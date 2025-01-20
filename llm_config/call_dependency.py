import os
import fnmatch
import pandas as pd


def find_jar_files(directory):
    jar_files = []

    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, '*.jar'):
            if not filename.endswith('tests.jar') and '/lib' not in root:
                jar_files.append(os.path.join(root, filename))

    return jar_files


def merge_to_csv(dependency_path):
    dataframes = []
    for filename in os.listdir(dependency_path):
        file_path = os.path.join(dependency_path, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            data = pd.read_csv(file, header=None, delimiter=' ', names=['Method', 'Called_Method'])
            data['Jar_name'] = os.path.splitext(filename)[0]
            dataframes.append(data)
    final_data = pd.concat(dataframes, ignore_index=True)
    final_data.to_csv(dependency_path + 'final.csv', index=False)

def single_jar_call_graph(jar_path, project_name):
    output_path = f'../data/call_graph/{project_name}/'
    output_name = output_path + jar_path.split('/')[-1][:-4] + '.txt'
    print(output_name)
    command = 'java -jar ' + '../data/java-callgraph/javacg-0.1-SNAPSHOT-static.jar ' + jar_path + ' > ' + output_name
    try:
        os.system(command)
        print('sucess')
    except Exception:
        print('fail')
    data = pd.read_csv(output_name, header=None, delimiter=' ', names=['Method', 'Called_Method'])
    data.to_csv(output_path + 'final.csv', index=False)
    # merge_to_csv(output_path)

def multi_jar_call_graph(project_directory, project_name):
    jar_paths = find_jar_files(project_directory)
    output_path = f'../data/call_graph/{project_name}/'
    for jar_path in jar_paths:
        output_name = output_path + jar_path.split('/')[-1][:-4] + '.txt'
        print(output_name)
        command = 'java -jar ' + '../data/java-callgraph/javacg-0.1-SNAPSHOT-static.jar ' + jar_path + ' > ' + output_name
        try:
            os.system(command)
            print('sucess')
        except Exception:
            print('fail')
    merge_to_csv(output_path)

#run following to get teh call graph
# jar_path = '../data/system/cassandra/build/apache-cassandra-4.0.5-SNAPSHOT.jar'
# project_name = 'cassandra'
# jar_path = '../data/system/h2-2.1.210-SNAPSHOT.jar'
# project_name = 'h2'
# single_jar_call_graph(jar_path, project_name)
# jar_path = '../data/system/dconverter/dconvert-1.0.0-alpha7.jar'
# project_name = 'dconverter'
# project_directory = '../data/system/DiagConfig/taint-analysis/subjectSys/prevayler/build/'
# project_name = 'prevayler'
# single_jar_call_graph(jar_path, project_name)
# multi_jar_call_graph(project_directory, project_name)
# project_name = 'batik'
# jar_path = '../data/system/s/batik/batik-all-1.14.jar'
# single_jar_call_graph(jar_path, project_name)
# project_directory = '../data/system/DiagConfig/taint-analysis/subjectSys/batik/'
# multi_jar_call_graph(project_directory, project_name)
# project_name = 'sunflow'
# jar_path = '../data/system/sunflow/sunflow.jar'
# single_jar_call_graph(jar_path, project_name)
# project_name = 'catena'
# jar_path = '../data/system/catena-0.0.1-SNAPSHOT.jar'
# single_jar_call_graph(jar_path, project_name)



