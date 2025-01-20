# ConfigSense

This repository contains the replication package for the paper: Leveraging LLM-Based Agents to Identify Performance-Sensitive Configurations in Software Systems

## Project Structure

The project directory includes the following main folders:

- **data/**: Contains the data required for the project.
  - **analysis_result/**: Stores the evaluation results for RQ1 and RQ2.
  - **baseline/**: Contains baseline data.
  - **method_context/**: Includes the code context for configurations across various systems.
  - **method_index/**: Contains method bodies and method signature indices for method retrieval purposes.

- **llm_config/**: Includes script files for running the experiments.
  - **requirements.txt**: Lists the dependencies needed for the project.
  - **main.py**: The main script to execute the project, you need to setup the openai api key and output/save path. 
  - **prompts.json**: Provides the prompt instructions for LLMs to perform the performance analysis task.
  - **langgraph.json**: Configuration file for LangGraph Studio setup. Note: you must set the OpenAI API key in the `.env` file.

- **rq2/**: Contains scripts for RQ2 experiments. You also need to setup the openai apikey and output/save path. 
