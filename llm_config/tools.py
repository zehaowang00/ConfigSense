import pandas as pd
from langchain_core.tools import tool
import json
import os
import re

@tool
def extract_configuration_context(target_system, config_name):
    """
    This function is to help to extract code context for a given configuration.
    target_system: str - The target system name.
    config_name: str - The configuration name.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = f'{current_dir}/data_new/method_context/{target_system}/{config_name}.csv'
    metric_context = pd.read_csv(file_path)
    code_context = metric_context['Method_body'].astype(str).str.cat(sep='\n')
    return code_context


@tool
def extract_method_body(target_system, method_signature):
    """
    This function is to help to retrieve method body from a given method signature.
    target_system: str - The target system name.
    method_signature: str - The method signature.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    method_index_path = f'{current_dir}/data_new/methods_index/{target_system}/methods_index.csv'
    methods_index_df = pd.read_csv(method_index_path)
    #matched_rows = methods_index_df[methods_index_df['Method Signature'].str.contains(method_signature, na=False)]

    # method_signatures = [
    #     re.escape(sig.strip().replace("`", "").replace("()", "").split('.')[-1])
    #     for sig in method_signature.split(',')
    # ]
    method_signatures = [
        sig.strip().replace("`", "").replace("()", "")
        for sig in method_signature.split(',')
    ]
    method_bodies = []

    for sig in method_signatures:

        if '.' in sig:  # If the signature includes class or package (ClassName.methodName)
            # Use str.contains for partial match including class/package name
            matched_rows = methods_index_df[methods_index_df['Method Signature'].str.contains(re.escape(sig), na=False, regex=True)]
            if matched_rows.empty:
                method_name_only = sig.split('.')[-1]
                matched_rows = methods_index_df[methods_index_df['Method Signature'].apply(lambda x: x.split('.')[-1] == method_name_only)]
        else:
            # Only match the method name if no class/package is included
            matched_rows = methods_index_df[methods_index_df['Method Signature'].apply(lambda x: x.split('.')[-1] == sig)]

        #matched_rows = methods_index_df[methods_index_df['Method Signature'].apply(lambda x: x.split('.')[-1] == sig)]

        if not matched_rows.empty:
            # Combine all matched method bodies into a single string
            #full_signature = matched_rows['Method Signature'].iloc[0]
            combined_method_body = "\n\n".join(matched_rows['Method Body'].values)
            method_body = f"Method Name: {sig}\nMethod Code:\n{combined_method_body}"
        else:
            method_body = f"Method Name: {sig}\nNo method body found for this signature."

        method_bodies.append(method_body)

    # Join all method bodies into a single string with clear formatting
    return "\n\n---\n\n".join(method_bodies)

