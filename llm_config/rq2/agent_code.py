from typing import List, Annotated, Sequence
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

# Define initial state structure
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    performance_sensitive: str
    code_context: str
    target_system: str
    config_name: str
    count: int

# Load prompts
with open('prompts.json', 'r') as f:
    prompts = json.load(f)
sensitivity_prompt = prompts["performance_sensitivity_prompt"]

# Define performance agent function without tools
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

# Initialize the language model
OPENAI_API_KEY = '' # only_code api key
#llm = ChatOpenAI(model="gpt-4o-2024-08-06")
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.3, api_key=OPENAI_API_KEY)

performance_agent = create_performance_agent(
    llm=llm,
    system_message=prompts["performance_agent"]["Role"] + " " + prompts["performance_agent"]["Primary Function"]
)

# Developer Code Extraction Node
def developer_extract_config_node(state):
    # Directly call the extract_configuration_context tool with target system and config name
    tool_response_content = extract_configuration_context.invoke({
        "target_system": state["target_system"],
        "config_name": state["config_name"]
    })

    new_message = HumanMessage(content=tool_response_content, sender="Developer")
    state["code_context"] = tool_response_content
    state["messages"] = [new_message]
    return state

# Performance Sensitivity Node
def performance_sensitivity_node(state, agent, prompt):
    prompt['configuration_name'] = state['config_name']
    prompt['code_context'] = state['code_context']
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
    return state

# Initialize model and memory
graph = StateGraph(AgentState)

# Add nodes to the graph
graph.add_node("DeveloperTool", developer_extract_config_node)
graph.add_node("PerformanceSensitivity", functools.partial(performance_sensitivity_node, agent=performance_agent, prompt=sensitivity_prompt))

# Define graph edges
graph.add_edge(START, "DeveloperTool")  # Initial edge to Developer Tool
graph.add_edge("DeveloperTool", "PerformanceSensitivity")  # Enter sensitivity analysis after code extraction
graph.add_edge("PerformanceSensitivity", END)  # End after sensitivity analysis

compiled_graph = graph.compile().with_config(run_name="Reduced System Analysis")

with open("graph_visualization.png", "wb") as f:
    f.write(compiled_graph.get_graph().draw_mermaid_png())
# Save results to JSON file
def save_to_json(state, system_name, config_name="default_config"):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    output_path = os.path.join(current_dir, f"data_new/analysis_result/{system_name}/{state['count']}/code/")
    output_file = os.path.join(output_path, f"{config_name}.json")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_data = {
        "performance_sensitive": state["performance_sensitive"],
        "config_name": config_name,
        "messages": [{"sender": msg.sender, "content": msg.content} for msg in state["messages"]]
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f,  indent=4)
        logging.info(f"{config_name} is processed and the results are saved.")

def load_configurations_list_for_systems(count, system_name):
    option_names = []
    option_path = f'./data_new/method_context/{system_name}/'
    for filename in os.listdir(option_path):
        if filename.endswith(".csv"):
            option_names.append(filename.split('.csv')[0])

    result_file_path = f'./data_new/analysis_result/{system_name}/{count}/code/'
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    option_names_processed = []
    for filename in os.listdir(result_file_path):
        if filename.endswith(".json"):
            option_names_processed.append(filename.split('.json')[0])
    option_names = list(set(option_names) - set(option_names_processed))
    return option_names

def setup_logging(output_path, count):
    log_dir = os.path.join(output_path, "log_code")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"{count}.txt")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file, mode='w'),
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
        "code_context":"", "performance_sensitive": "No",
        "target_system": system_name, "config_name": config_name,
        "count": count
    }

    result_state = compiled_graph.invoke(initial_state, config={"recursion_limit": 40})
    save_to_json(result_state, system_name, config_name=config_name)

def main():
    system_names = ['h2', 'dconverter','catena','batik','sunflow','prevayler', 'cassandra']
    run_times = 5  # this number is about how many times you want to run the graph
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
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        logging.info(f"------------ System {system_name} processing completed in {elapsed_time:.2f} minutes. -------------------")
if __name__ == '__main__':
    main()