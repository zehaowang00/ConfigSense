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
import sys
from IPython.display import Image, display

sys.path.append('../llm_config')

class IgnoreHTTPRequestsFilter(logging.Filter):
    def filter(self, record):
        return "HTTP Request" not in record.getMessage()


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    performance_sensitive: str
    raw_code_context: str
    code_context: str
    target_system: str
    config_name: str
    count: int


tools = [extract_configuration_context, extract_method_body]
tool_node = ToolNode(tools)

with open('prompts.json', 'r') as f:
    prompts = json.load(f)
summary_prompt = prompts["developer_context_summary_prompt"]
filter_prompt = prompts['developer_filter_context_prompt']
sensitivity_prompt = prompts["performance_sensitivity_prompt"]


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
    prompt = prompt.partial(system_message=system_message)
    return prompt | llm


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


OPENAI_API_KEY = '' # api key code dev
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
    tool_response_content = extract_configuration_context.invoke({
        "target_system": state["target_system"],
        "config_name": state["config_name"]
    })
    new_message = HumanMessage(content=tool_response_content, sender="Developer")
    state["raw_code_context"] = tool_response_content
    state["messages"] = [new_message]
    return state


def developer_generate_summary_node(state, agent, prompt):
    prompt['configuration_name'] = state['config_name']
    prompt_str = json.dumps(prompt)
    summary_request_message = state["messages"] + [HumanMessage(
        content=f"{prompt_str}\n\nHere is the code context to summarize for system understanding:\n{state['code_context']}",
        sender="Developer"
    )]
    response = agent.invoke({"messages": summary_request_message})
    response_content = response.content if isinstance(response, AIMessage) else str(response)
    summary_message = AIMessage(content=f"Summary of the extracted code context: {response_content}",
                                sender="Developer summary")
    state["messages"] = [summary_message]
    state["code_context"] += f"\n\n--- Summary by Developer ---\n{response_content.strip()}"
    return state


def developer_filter_context_node(state, agent, prompt):
    prompt['configuration_name'] = state['config_name']
    prompt['raw_code_context'] = state['raw_code_context']
    prompt_str = json.dumps(prompt)
    filter_request_message = state["messages"] + [HumanMessage(
        content=f"{prompt_str}\n\nPlease analyze the provided raw code context and return only the segments directly related to the configuration '{state['config_name']}'.",
        sender="Developer"
    )]
    response = agent.invoke({"messages": filter_request_message})
    filtered_code_context = response.content if isinstance(response, AIMessage) else str(response)
    filter_message = AIMessage(content=f"Filter of the code context that is related to configuration: {filtered_code_context}",
                                sender="Developer Filter")
    state["messages"] = [filter_message]
    state["code_context"] = filtered_code_context.strip()
    return state


def performance_sensitivity_node(state, agent, prompt):
    prompt['configuration_name'] = state['config_name']
    prompt['code_context'] = state['code_context']
    human_message = HumanMessage(content=json.dumps(prompt), sender="performance agent sensitivity")
    response = agent.invoke({"messages": state["messages"] + [human_message]})
    response_content = response.content if isinstance(response, AIMessage) else str(response)
    ai_message = AIMessage(content=response_content, sender="PerformanceSensitivity")
    state["messages"] = [human_message, ai_message]
    state["performance_sensitive"] = "Yes" if "performance sensitive" in response_content.lower() else "No"
    return state


graph = StateGraph(AgentState)

developer_tool_node = functools.partial(developer_extract_config_node)
developer_summary_node = functools.partial(developer_generate_summary_node, prompt=summary_prompt, agent=developer_agent)
developer_filter_context_node = functools.partial(developer_filter_context_node, prompt=filter_prompt, agent=developer_agent)
performance_sensitivity_node = functools.partial(performance_sensitivity_node, agent=performance_agent, prompt=sensitivity_prompt)

graph.add_node("DeveloperTool", developer_tool_node)
graph.add_node("DeveloperFilterContext", developer_filter_context_node)
graph.add_node("DeveloperSummary", developer_summary_node)
graph.add_node("PerformanceSensitivity", performance_sensitivity_node)

graph.add_edge(START, "DeveloperTool")
graph.add_edge("DeveloperTool", "DeveloperFilterContext")
graph.add_edge("DeveloperFilterContext", "DeveloperSummary")
graph.add_edge("DeveloperSummary", "PerformanceSensitivity")
graph.add_edge("PerformanceSensitivity", END)

compiled_graph = graph.compile().with_config(run_name="Code Context Analysis")

with open("graph_visualization.png", "wb") as f:
    f.write(compiled_graph.get_graph().draw_mermaid_png())

def save_to_json(state, system_name, config_name="default_config"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, f"data_new/analysis_result/{system_name}/{state['count']}/code_dev/")
    output_file = os.path.join(output_path, f"{config_name}.json")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_data = {
        "performance_sensitive": state["performance_sensitive"],
        "config_name": config_name,
        "messages": [{"sender": msg.sender, "content": msg.content} for msg in state["messages"]]
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        logging.info(f"{config_name} is processed and the results are saved.")


def load_configurations_list_for_systems(count, system_name):
    option_names = []
    option_path = f'./data_new/method_context/{system_name}/'
    for filename in os.listdir(option_path):
        if filename.endswith(".csv"):
            option_names.append(filename.split('.csv')[0])
    result_file_path = f'./data_new/analysis_result/{system_name}/{count}/code_dev/'
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)
    option_names_processed = [filename.split('.json')[0] for filename in os.listdir(result_file_path) if filename.endswith(".json")]
    return list(set(option_names) - set(option_names_processed))


def setup_logging(output_path, count):
    log_dir = os.path.join(output_path, "log_code_dev")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
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


def run_graph(system_name, config_name, count):
    initial_state = {
        "messages": [
            HumanMessage(content="Initialize the analysis for the target system and configuration.", sender="System")],
        "raw_code_context": "", "code_context": "", "performance_sensitive": "No",
        "target_system": system_name, "config_name": config_name, "count": count
    }
    result_state = compiled_graph.invoke(initial_state, config={"recursion_limit": 40})
    save_to_json(result_state, system_name, config_name=config_name)


def main():
    system_names = ['h2', 'dconverter', 'catena', 'batik', 'sunflow', 'prevayler', 'cassandra']
    run_times = 5  # This number represents how many times you want to run the graph

    for system_name in system_names:
        start_time = time.time()
        for count in range(1, run_times + 1):
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       f"data_new/analysis_result/{system_name}/{count}")
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
        logging.info(
            f"------------ System {system_name} processing completed in {elapsed_time:.2f} minutes. -------------------")


if __name__ == '__main__':
    main()