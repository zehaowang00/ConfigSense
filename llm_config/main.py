from typing import List, Annotated, Sequence

from openai import api_key
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
import functools
from tools import extract_configuration_context, extract_method_body
import json
import operator
import os
import time
import logging
import requests
import sys
from IPython.display import Image, display
import pandas as pd


sys.path.append('../llm_config')

class IgnoreHTTPRequestsFilter(logging.Filter):
    def filter(self, record):
        return "HTTP Request" not in record.getMessage()

# Define initial state structure
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    clarity: bool
    performance_sensitive: str
    raw_code_context: str
    code_context: str
    unclear_method_signature:str
    target_system: str
    config_name: str
    unclear_code_summary: bool
    reflection_count: int
    unclear_analysis_count: int
    reflection_needed: bool
    count: int

# Define tools
tools = [extract_configuration_context, extract_method_body]
tool_node = ToolNode(tools)

with open('prompts.json', 'r') as f:
    prompts = json.load(f)
clarity_prompt = prompts["performance_clarity_prompt"]
reflection_prompt = prompts["reflection_prompt"]
sensitivity_prompt = prompts["performance_sensitivity_prompt"]
summary_prompt = prompts["developer_context_summary_prompt"]
filter_prompt = prompts['developer_filter_context_prompt']

# General agent creation function
def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an agent equipped with tools to assist with tasks.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

def create_developer_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an agent to assist with tasks.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    # prompt = prompt.partial(system_message=system_message)
    # prompt = prompt.partial(tool_names=", ".join([getattr(tool, "name", "Unnamed Tool") for tool in tools]))  # Handle tool names
    # return prompt
    prompt = prompt.partial(system_message=system_message)
    return prompt | llm


# Performance agent function without tools
def create_performance_agent(llm, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an agent to assist with tasks.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    return prompt | llm

# Create agents
OPENAI_API_KEY = ''
#llm = ChatOpenAI(model="gpt-4o-2024-08-06")
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.3, api_key=OPENAI_API_KEY)
developer_agent = create_developer_agent(
    llm=llm,
    tools=[extract_configuration_context, extract_method_body],
    system_message=prompts["developer_agent"]["Role"] + " " + prompts["developer_agent"]["Primary Function"]
)

performance_agent = create_performance_agent(
    llm=llm,
    system_message=prompts["performance_agent"]["Role"] + " " + prompts["performance_agent"]["Primary Function"]
)

def developer_extract_config_node(state):
    # Directly call the extract_configuration_context tool with target system and config name
    # if state.get("developer_tool_executed", False):
    #     return state

        # 
    tool_response_content = extract_configuration_context.invoke({
        "target_system": state["target_system"],
        "config_name": state["config_name"]
    })

    new_message = HumanMessage(content=tool_response_content, sender="Developer")
    #state["raw_code_context"] = tool_response_content
    state["raw_code_context"] = tool_response_content
    state['unclear_code_summary'] = False
    # At the end, add only the new messages from this node to state["messages"]
    #state["messages"].append(new_message)
    state["messages"] = [new_message]
    #return {"messages": [new_message], "sender": "Developer"}
    return state

def developer_extract_method_node(state):

    method_signature = state.get("unclear_method_signature")
    if not method_signature:
        new_message_content = (
            "The request does not specify any method names to extract, making it impossible to proceed.\n"
        )
        new_message = HumanMessage(content=new_message_content, sender="Developer")
        state["messages"] = [new_message]
        state['unclear_code_summary'] = True
        return state

    tool_response_content = extract_method_body.invoke({
        "target_system": state["target_system"],
        "method_signature": method_signature
    })
    new_message_content = (
        f"The following methods were found to be unclear in the performance analysis context.\n"
        f"Here is the code context to aid further examination:\n\n{tool_response_content}"
    )
    new_message = HumanMessage(content=new_message_content, sender="Developer")

    # Append only the new message to state["code_context"] for ongoing reference
    state["code_context"] += f"\n\n{new_message_content}"
    state["messages"] = [new_message]
    state['unclear_code_summary'] = True
    return state

def developer_generate_summary_node(state, agent, prompt):
    prompt['configuration_name'] = state['config_name']
    prompt_str = json.dumps(prompt)
    # Create the request message based on the updated functional context of the configuration
    if state.get("unclear_code_summary", False):
        summary_request_message = state["messages"] + [HumanMessage(
            content=f"{prompt_str}\n\nSome unclear method is added to the context, here is the code context to summarize for system understanding:\n{state['code_context']}. ",
            sender="Developer"
        )]
    else:
        summary_request_message = state["messages"]
        state['unclear_code_summary'] = False
    response = agent.invoke({"messages": summary_request_message})
    #response = agent.invoke({"messages": [summary_request_message]})
    # Construct the summary as a new message with the response content
    # response_content = response.get("content", "") if isinstance(response, dict) else str(response)
    if isinstance(response, AIMessage):
        response_content = response.content
    elif isinstance(response, dict) and "content" in response:
        response_content = response["content"]
    else:
        response_content = str(response)
    summary_message = AIMessage(content=f"Summary of the extracted code context: {response_content}",
                                sender="Developer summary")

    # Replace state['messages'] with only the summary request and its response
    #state["messages"] = [summary_request_message, summary_message]
    state["messages"] = [summary_message]
    # Update state with the summary in the 'code_context' for downstream use
    state["code_context"] += f"\n\n--- Summary by Developer ---\n{response_content.strip()}"

    return state


def developer_filter_context_node(state, agent, prompt):
    # Prepare prompt with unfiltered code context for agent analysis
    prompt['configuration_name'] = state['config_name']
    prompt['raw_code_context'] = state['raw_code_context']
    prompt_str = json.dumps(prompt)

    # Create the message asking the agent to filter the relevant configuration-related code
    filter_request_message = state["messages"] + [HumanMessage(
        content=f"{prompt_str}\n\nPlease analyze the provided raw code context and return only the segments directly related to the configuration '{state['config_name']}'. Focus on configuration-specific code, omitting test or unrelated segments to provide a clear, concise context.",
        sender="Developer"
    )]

    # Invoke the agent to perform the filtering and receive the response
    response = agent.invoke({"messages": filter_request_message})

    # Extract the filtered content from the agent's response
    if isinstance(response, AIMessage):
        filtered_code_context = response.content
    elif isinstance(response, dict) and "content" in response:
        filtered_code_context = response["content"]
    else:
        filtered_code_context = str(response)

    filter_message = AIMessage(content=f"Filter of the code context that is related to configuration: {filtered_code_context}",
                                sender="Developer Filter")
    state["messages"] = [filter_message]
    # Update the state with the filtered context as returned by the agent
    state["code_context"] = filtered_code_context.strip()

    return state


# Clarity Analysis Node
def clarity_analysis_node(state, agent, prompt):
    prompt['configuration_name'] = state['config_name']
    prompt['code_context'] = state['code_context']
    if not state.get('clarity') and len(state.get('unclear_method_signature', '')) > 0:

        if state['unclear_analysis_count'] >= 1:
            prompt['Reminder'] = (
                "This is a follow-up review. Focus on previously unresolved methods or code elements "
                "to determine if they now have adequate context for performance analysis."
            )

            #prompt['Unclear code and the corresponding summary'] = state['messages'][-1].content
            prompt['The AI agent has analyzed the unclear method name'] = state['unclear_method_signature']
            prompt['Note'] = (
            "The AI agent has analyzed the unclear method: "
            f"{state['unclear_method_signature']} and provided your requested information for further performance analysis."
            "You must remember that you cannot ask the AI agent to analyze the same above methods again, no more information about the same method can be provided."
            )
    human_message = HumanMessage(content=json.dumps(prompt), sender="performance agent Clarity Analysis")
    response = agent.invoke({"messages": state["messages"] + [human_message]})
    if isinstance(response, AIMessage):
        response_content = response.content
    elif isinstance(response, dict) and "content" in response:
        response_content = response["content"]
    else:
        response_content = str(response)

    ai_message = AIMessage(content=response_content, sender="ClarityAnalysis")

    if "clear for conducting performance analysis" in response_content.lower():
        state["clarity"] = True
    elif "unclear method" in response_content.lower():
        state["clarity"] = False
        unclear_methods = [
            line.split(":")[-1].strip()
            for line in response_content.splitlines()
            if "unclear method" in line.lower()
        ]
        current_methods = [method for method in state.get("unclear_method_signature", "").split(", ") if method]
        for method in unclear_methods:
            if method not in current_methods:
                current_methods.append(method)
        state["unclear_method_signature"] = ", ".join(current_methods)

    state['unclear_analysis_count'] = state.get('unclear_analysis_count', 0) + 1
    state["messages"] = [human_message, ai_message]
    return state


# Performance Sensitivity Node
def performance_sensitivity_node(state, agent, prompt):
    prompt['configuration_name'] = state['config_name']
    prompt['code_context'] = state['code_context']
    if state.get('reflection_needed'):
        prompt['Reminder'] = (
            "This is a follow-up sensitivity analysis. Re-evaluate the configuration, focusing on unresolved points "
            "from previous steps."
        )
        prompt['Message from previous performance analysis'] = state['messages'][-1].content
        prompt['Note'] = "Please review the previous message for further performance sensitivity analysis."
    human_message = HumanMessage(content=json.dumps(prompt), sender="performance agent sensitivity")
    response = agent.invoke({"messages": state["messages"] + [human_message]})

    if isinstance(response, AIMessage):
        response_content = response.content
    elif isinstance(response, dict) and "content" in response:
        response_content = response["content"]
    else:
        response_content = str(response)

    ai_message = AIMessage(content=response_content, sender="PerformanceSensitivity")
    state["messages"] = [ai_message]

    if "performance sensitive" in response_content.lower():
        state["performance_sensitive"] = "Yes"
    elif "performance insensitive" in response_content.lower():
        state["performance_sensitive"] = "No"
    state["messages"] = [human_message, ai_message]
    return state


# Reflection Node
def reflection_node(state, agent, prompt):
    prompt['configuration_name'] = state['config_name']
    if state.get('reflection_needed'):
        prompt['Reminder'] = (
            "This is a follow-up reflection. Please review any unresolved issues from the previous performance analysis "
            "when verifying the sensitivity conclusions."
        )
        prompt['Message from previous performance analysis'] = state['messages'][-1].content
        prompt['Note'] = "The AI agent has provided suggestions for further performance analysis."
    human_message = HumanMessage(content=json.dumps(prompt), sender="performance agent reflection")
    response = agent.invoke({"messages": state["messages"] + [human_message]})
    if isinstance(response, AIMessage):
        response_content = response.content
    elif isinstance(response, dict) and "content" in response:
        response_content = response["content"]
    else:
        response_content = str(response)

    ai_message = AIMessage(content=response_content, sender="Reflection")

    state["reflection_needed"] = "reflection_needed" in response_content.lower()
    if state["reflection_needed"]:
        state["reflection_count"] = state.get("reflection_count", 0) + 1
    state["messages"] = [human_message, ai_message]
    return state

# Developer router
def router_performance_clarity(state):
    if state.get("unclear_analysis_count", 0) >= 3:
        return "PerformanceSensitivity"
    if not state["clarity"]:
        return "DeveloperMethodBody"
    return "PerformanceSensitivity"

def router_performance_sensitivity(state):
    return "Reflection"

def router_reflection(state):
    # If the reflection count reaches 3, terminate the cycle
    if state.get("reflection_count", 0) >= 3:
        return END

    # If reflection is needed and within allowed attempts, return to PerformanceSensitivity
    if state.get("reflection_needed", False):
        return "PerformanceSensitivity"

    return END

# Initialize model and memory
#memory = MemorySaver()
graph = StateGraph(AgentState)

# Create node variables
developer_tool_node = functools.partial(developer_extract_config_node)
developer_method_body_node = functools.partial(developer_extract_method_node)
developer_summary_node = functools.partial(developer_generate_summary_node, prompt=summary_prompt, agent=developer_agent)
developer_filter_context_node = functools.partial(developer_filter_context_node, prompt=filter_prompt, agent=developer_agent)
clarity_analysis_node = functools.partial(clarity_analysis_node, agent=performance_agent, prompt=clarity_prompt)
performance_sensitivity_node = functools.partial(performance_sensitivity_node, agent=performance_agent, prompt=sensitivity_prompt)
reflection_node = functools.partial(reflection_node, agent=performance_agent, prompt=reflection_prompt)

# Add nodes to the graph
graph.add_node("DeveloperTool", developer_tool_node)
graph.add_node("DeveloperFilterContext", developer_filter_context_node)
graph.add_node("DeveloperMethodBody", developer_method_body_node)
graph.add_node("DeveloperSummary", developer_summary_node)
graph.add_node("PerformanceClarity", clarity_analysis_node)
graph.add_node("PerformanceSensitivity", performance_sensitivity_node)
graph.add_node("Reflection", reflection_node)

# Define graph edges
graph.add_edge(START, "DeveloperTool")  # Initial edge to Developer Tool
graph.add_edge("DeveloperTool", "DeveloperFilterContext")  # Enter filtering after initial extraction
graph.add_edge("DeveloperFilterContext", "DeveloperSummary")  # Enter summary generation after filtering
# graph.add_edge("DeveloperTool", "DeveloperSummary")  # Enter filtering after initial extraction
graph.add_edge("DeveloperSummary", "PerformanceClarity")  # Enter clarity analysis after summary
graph.add_conditional_edges("PerformanceClarity", router_performance_clarity, {
    "DeveloperMethodBody": "DeveloperMethodBody",  # If unclear, return to Developer Tool for further extraction
    "PerformanceSensitivity": "PerformanceSensitivity"  # If clear, proceed to performance sensitivity analysis
})
graph.add_edge("DeveloperMethodBody", "DeveloperSummary")
graph.add_conditional_edges("PerformanceSensitivity", router_performance_sensitivity, {
    "Reflection": "Reflection"  # Enter reflection after analysis
})
graph.add_conditional_edges("Reflection", router_reflection, {
    "PerformanceSensitivity": "PerformanceSensitivity",  # If reflection is inadequate, return for re-analysis
    END: END  # Single endpoint: analysis complete without issues
})

compiled_graph = graph.compile().with_config(run_name="Complete System Analysis")

# Visualize and display the graph
with open("graph_visualization.png", "wb") as f:
    f.write(compiled_graph.get_graph().draw_mermaid_png())

# Save results to JSON file
def save_to_json(state, system_name, config_name="default_config"):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    output_path = os.path.join(current_dir, f"data_new/analysis_result/{system_name}/{state['count']}/base/")
    output_file = os.path.join(output_path, f"{config_name}.json")
    # Ensure the directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_data = {
        "performance_sensitive": state["performance_sensitive"],
        "config_name": config_name,
        "messages": [{"sender": msg.sender, "content": msg.content} for msg in state["messages"]]
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f,  indent=4)
        #print(f"{config_name} is processed and the results are saved.")
        logging.info(f"{config_name} is processed and the results are saved.")

def load_configurations_list_for_systems(count, system_name):
    option_names = []
    option_path = f'./data_new/method_context/{system_name}/'
    for filename in os.listdir(option_path):
        if filename.endswith(".csv"):
            option_names.append(filename.split('.csv')[0])

    result_file_path = f'./data_new/analysis_result/{system_name}/{count}/base/'
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    option_names_processed = []
    for filename in os.listdir(result_file_path):
        if filename.endswith(".json"):
            option_names_processed.append(filename.split('.json')[0])
    option_names = list(set(option_names) - set(option_names_processed))
    return option_names


def setup_logging(output_path, count):
    # Create log directory if it doesn't exist
    log_dir = os.path.join(output_path, "log_base")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up log file for the current run
    log_file = os.path.join(log_dir, f"{count}.txt")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    for handler in logging.getLogger().handlers:
        handler.addFilter(IgnoreHTTPRequestsFilter())
    return log_file

# Run the graph and save state
def run_graph(system_name, config_name, count):
    initial_state = {
        "messages": [
            HumanMessage(content="Initialize the analysis for the target system and configuration.", sender="System")],
        "raw_code_context": "", "code_context":"", "clarity": False, "performance_sensitive": "No",
        "unclear_code_summary": False, "unclear_method_signature": "", "target_system": system_name, "config_name": config_name,
        "reflection_count": 0, "unclear_analysis_count": 0, "reflection_needed": False, "count": count
    }
    thread = {"configurable": {"thread_id": "1"}}  # Define thread ID to manage context
    
    result_state = compiled_graph.invoke(initial_state, config={"recursion_limit": 40})  # Pass thread info to graph call
    save_to_json(result_state, system_name, config_name=config_name)  # Save results to JSON file

def main():
    system_names = ['h2', 'dconverter','catena','batik','sunflow','prevayler', 'cassandra']
    #system_names = ['cassandra']
    run_times = 5 # this number is about how many times you want to run the graph
    for system_name in system_names:
        start_time = time.time()
        for count in range(1, run_times + 1):
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data_new/analysis_result/{system_name}/{count}")
            setup_logging(output_path, count)
            for config in load_configurations_list_for_systems(count, system_name):
                config_name = config
                try:
                    run_graph(system_name, config_name, count)
                except Exception as e:
                    logging.error(f"Error processing {config_name}: {e}")
            logging.info(f"---------- {count} run completed for system {system_name} ----------------")
        end_time = time.time()  # Record the end time for each system
        elapsed_time = (end_time - start_time) / 60  # Convert elapsed time to minutes
        logging.info(f"------------ System {system_name} processing completed in {elapsed_time:.2f} minutes. -------------------")
if __name__ == '__main__':
    main()

