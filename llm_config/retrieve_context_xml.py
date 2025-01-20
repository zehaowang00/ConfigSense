import pandas as pd
import os
import ast
import subprocess
# from javalang.parse import parse
# from javalang.tree import MethodDeclaration
# import javalang as jl
#import xml.etree.ElementTree as ET
from lxml import etree as ET
import re


def extract_method_name(full_name):
    return full_name.split(':')[-1].split('(')[0]


def get_method_file_path(directory, method):
    class_name = ''
    class_path = ''
    if method.count(':') == 2:
        class_path = method.split(':')[1].replace('.', '/')
        class_name = method.split(':')[1].split('.')[-1]
        if '$' in class_path:
            class_path = class_path.split('$')[0]
            class_name = method.split(':')[1].split('.')[-1]
            if '$' in class_name:
                class_name = class_name.split('$')[0]
    else:
        class_path = method.split(':')[0].split(')')[1].replace('.', '/')
        class_name = method.split(':')[0].split('.')[-1]
        if '$' in class_name:
            class_name = class_name.split('$')[0]
            
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == class_name + '.java' and '/'.join(class_path.split('/')[:-1]) in root:
                path = os.path.join(root, file)
                return path


def generate_xml_fle(file_path, project, xml_output_directory):
    file_name = file_path.split('/')[-1].split('.')[0]
    xml_path = xml_output_directory + project + '/' + file_name + ".xml"
    command_to_xml = 'srcml ' + file_path + ' -o ' + xml_path
    os.system(command_to_xml)
    return xml_path

def extract_source(srcml_file, method_name, class_name, full_method_name):
    # Load the srcML file
    try:
        tree = ET.parse(srcml_file)
        root = tree.getroot()
        result = 'not found'
        # Define the namespace to find elements
        ns = {'src': 'http://www.srcML.org/srcML/src'}
        if '<init>' in method_name:
            if has_inner_element(class_name):
                search_target = get_target_name(class_name)
                result = extract_body(root, ns, '*[src:name]', search_target, full_method_name)
            else:
                result = extract_body(root, ns, 'constructor', class_name, full_method_name)
        # if 'build' in method_name:
        #     search_target = 'Benchmark'
        #     result = extract_body(root, ns, '*[src:name]', search_target, full_method_name)
        elif '<clinit>' in method_name:
            if has_inner_element(class_name):
                search_target = get_target_name(class_name)
                result = extract_body(root, ns, '*[src:name]', search_target, full_method_name)
            else:
                result = extract_body(root, ns, 'class', class_name, full_method_name)
        elif '$' in method_name:
            if has_inner_element(class_name):
                search_target = get_target_name(class_name)
                result = extract_body(root, ns, '*[src:name]', search_target, full_method_name)
            else:
                result = extract_body(root, ns, 'class', class_name, full_method_name)
        else:
            result = extract_body(root, ns, 'function', method_name, full_method_name)
        return result
    except Exception as e:
        return "not found"
    # Search for the function/method within the XML structure


def has_inner_element(class_name):
    if '$' in class_name:
        return True
    else:
        return False


def get_target_name(class_name):
    start = class_name.split('$')[0]
    end = class_name.split('$')[1]
    if end.isdigit():
        return start
    elif end[0].isdigit():
        #return end[1:]
        return start
    else:
        #return end
        return start


def extract_body(root, ns, element_type, search_name, full_method_name):
    function_texts = []
    parameters = full_method_name.split('(')[1].split(')')[0]
    if parameters:
        parameters = [item.split('.')[-1] for item in parameters.split(',')]
    else:
        parameters = []
    matching_function_count = 0
    for function in root.findall(f".//src:{element_type}", ns):
        name_element = function.find("src:name", ns)
        if name_element is not None:
            function_name = name_element.text
            if function_name == search_name:
                matching_function_count += 1
    for function in root.findall(f".//src:{element_type}", ns):
        name = function.find("src:name", ns)
        if matching_function_count > 1:
            parameters = [parameter.split('$')[-1] if '$' in parameter else parameter for parameter in parameters]
            function_text = ET.tostring(function, encoding='unicode', method='text')
            pattern = r'\b\w+\s*\(([\s\S]*?)\)\s*(?:throws [\w, \s]*)?\{'
            match = re.search(pattern, function_text)
            if match:
                tmp_parameters = ' '.join(match.group(1).split())
                extracted_params = tmp_parameters.split(',')
            else:
                extracted_params = []
            if name is not None and name.text == search_name and all(param in function_text for param in parameters):
                comments = []
                sibling = function.getprevious()
                while sibling is not None and sibling.tag.endswith('comment'):
                    comment_text = "".join(sibling.itertext())
                    comments.append(comment_text.strip())
                    sibling = sibling.getprevious()
                comments.reverse()
                function_text = ET.tostring(function, encoding='unicode', method='text')
                if len(comments) > 0:
                    function_texts.append("\n".join(comments) + '\n' + function_text)
                else:
                    function_texts.append(function_text)
        else:
            if name is not None and name.text == search_name:
                comments = []
                sibling = function.getprevious()
                while sibling is not None and sibling.tag.endswith('comment'):
                    comment_text = "".join(sibling.itertext())
                    comments.append(comment_text.strip())
                    sibling = sibling.getprevious()
                comments.reverse()
                function_text = ET.tostring(function, encoding='unicode', method='text')
                if len(comments) > 0:
                    function_texts.append("\n".join(comments) + '\n' + function_text)
                else:
                    function_texts.append(function_text)
    if len(function_texts) == 0:
        return 'not found'
    else:
        return "\n".join(function_texts)

def extract_class_name(method):
    class_name = ''
    if method.count(':') == 2:
        class_name = method.split(':')[1].split('.')[-1]
    else:
        class_name = method.split(':')[0].split('.')[-1]
    return class_name

#projects = ['prevayler']
#projects = ['dconverter']
#projects = ['cassandra']
#projects = ['catena']
#projects = ['batik']
#projects = ['sunflow']
projects = ['h2']
project_path = {
                'cassandra':'../data/system/cassandra/',
                'h2':'../system/h2database/h2/',
                'dconverter':'../data/system/density-converter/',
                'prevayler':'../data/system/prevayler/',
                'batik': '../data/system/xmlgraphics-batik/',
                'sunflow': '../data/system/sunflow/',
                'catena': '../data/system/catena2/'
                }
for project in projects:
    xml_path = '../data/xml/'
    directory = project_path[project]
    metric_output_directory = '../data/method_context/'
    #output_directory_csv = output_directory + 'csv/'
    option_file_path = '../config/location/' + f'{project}_location.csv'
    option_df = pd.read_csv(option_file_path)
    #option_df.to_csv('../config/' + project + '/full_location_tmp2.csv', index=False)
    option_df = option_df.drop_duplicates()
    option_df['Method_short'] = option_df['Method'].apply(extract_method_name)
    option_df['path'] = option_df['Method'].apply(lambda x: get_method_file_path(directory, x))
    condition = option_df[
                    'path'] != '../system/cassandra/src/java/org/apache/cassandra/net/VerbTimeouts.java'
    option_df = option_df[condition]
    option_df.dropna(subset=['path'], inplace=True)
    option_df['class_name'] = option_df['Method'].apply(lambda x: extract_class_name(x))
    option_df.to_csv('../config/' + project + '/full_location_tmp.csv', index=False)
    option_df['xml_path'] = option_df['path'].apply(lambda x: generate_xml_fle(x, project, xml_path))
    option_df['Method_body'] = option_df.apply(lambda row: extract_source(row['xml_path'], row['Method_short'], row['class_name'], row['Method']), axis=1)
    #option_df['Method_body'] = option_df.apply(lambda row: filter_methods(x['Method_body'], x['Called_Method']))
    option_df.to_csv('../config/' + project + '/full_location_pure.csv', index=False)
    print(option_df['option'].unique().tolist())
    unique = option_df['option'].unique().tolist()

    for option in unique:
        specific_option_df = option_df[option_df['option'] == option]
        option = str(option)
        print(option)
        specific_option_df.to_csv(metric_output_directory + project + '/' + option + '.csv', index=False)

