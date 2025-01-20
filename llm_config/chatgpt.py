from openai import OpenAI
import os
import json

def load_api_key(key_path):
    with open(key_path, 'r') as file:
        data = json.load(file)
        return data['Key']

def get_completion(client, prompt):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return response.choices[0].message.content

def load_configurations_list_for_systems(count, system_name):
    option_names = []
    option_path = f'../llm_config/data_new/method_context/{system_name}/'
    for filename in os.listdir(option_path):
        if filename.endswith(".csv"):
            option_names.append(filename.split('.csv')[0])

    result_file_path = f'../llm_config/data_new/analysis_result/{system_name}/{count}/chatgpt/'
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    option_names_processed = []
    for filename in os.listdir(result_file_path):
        if filename.endswith(".json"):
            option_names_processed.append(filename.split('.json')[0])
    option_names = list(set(option_names) - set(option_names_processed))
    return option_names


projects = ['h2', 'dconverter', 'catena', 'batik', 'sunflow', 'prevayler', 'cassandra']
api_key_path = "/Users/wang/Documents/project/api_key.json"
api_key = load_api_key(api_key_path)
client = OpenAI(api_key=api_key)

for project_name in projects:
    for index in range(1, 6):
        option_names = load_configurations_list_for_systems(index, project_name)
        for config_name in option_names:
            save_file_path = f'/Users/wang/Documents/project/configuration_code_understanding/code4/llm_config/data_new/analysis_result/{project_name}/{index}/chatgpt/' + config_name + '.json'
            prompt = {
                "Role": "You are a professional performance engineer, you are required to classify the performance sensitivity of the configuration.",
                "Requirement": "Determine if the given software configuration is performance-sensitive based on the impact of its "
                               "value change on system performance. A configuration is deemed performance-sensitive if its value "
                               "change significantly affects system performance, such as causing notable variations in execution "
                               "time or memory consumption.",
                "Output format requirement": "Answer the question, output the JSON format, the keys must be 'performance_sensitive', 'reason'.",
                "Name of the configuration requested to be analyzed": config_name,
                "software project name": project_name,
                "Question": "1. Is the configuration option performance-sensitive? (classification_result should be Yes or No.)"
                            "2. What is the reason for your classification? (Explain reason from the system level)"
            }
            response = get_completion(client, json.dumps(prompt))
            # Parse the response once
            response_data = json.loads(response)  # Decode the JSON string only once

            # Assign the fields directly from the response
            output_data = {
                "performance_sensitive": response_data.get("performance_sensitive"),
                "reason": response_data.get("reason")
            }
            print(f"Finished processing {config_name} for run {index}")
            with open(save_file_path, 'w') as file:
                json.dump(output_data, file, indent=4)
        print(f'Finished project {project_name} for run {index}')