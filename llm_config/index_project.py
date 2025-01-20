import os
import javalang
import pandas as pd

def build_methods_index(project_root, method_name=None):
    """
    Build an index of methods with signatures and bodies (including Javadoc).
    If a method_name is provided, it will only return that specific method's body (including Javadoc).
    """
    methods = []
    for subdir, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(subdir, file)
                class_name = os.path.splitext(os.path.basename(file))[0]  # Extract class name from file name
                with open(file_path, 'r', encoding='utf-8') as f:
                    codelines = f.readlines()
                    code_text = ''.join(codelines)
                    try:
                        tree = javalang.parse.parse(code_text)
                        package_name = tree.package.name if tree.package else ""
                        for _, method_node in tree.filter(javalang.tree.MethodDeclaration):
                            start_pos, end_pos, start_line, end_line = get_method_start_end(tree, method_node)
                            method_text, start_line, end_line, last_end_line_index = get_method_text(
                                start_pos, end_pos, start_line, end_line, None, codelines)
                            javadoc = get_javadoc_comments(start_line, codelines)

                            # Append Javadoc to the method body
                            method_with_javadoc = javadoc + "\n" + method_text if javadoc else method_text

                            # Include class name in the signature
                            signature = f"{package_name}.{class_name}.{method_node.name}"

                            # If method_name is provided, only extract the specified method
                            if method_name and method_node.name == method_name:
                                print(f"Extracted method: {signature}")
                                return pd.DataFrame([[signature, method_with_javadoc]],
                                                    columns=["Method Signature", "Method Body"])

                            # If no specific method_name is provided, add all methods to the list
                            methods.append([signature, method_with_javadoc])

                    except javalang.parser.JavaSyntaxError as e:
                        print(f"Syntax error in file: {file_path} - {e}")
                    except Exception as e:
                        print(f"Error processing file: {file_path} - {e}")

    return pd.DataFrame(methods, columns=["Method Signature", "Method Body"])


def get_javadoc_comments(start_line, codelines):
    """
    Extract the Javadoc comment that precedes a method definition.
    It assumes the Javadoc is located right before the method definition.
    """
    javadoc = []
    for i in range(start_line - 2, -1, -1):  # Look backwards from the line before method definition
        line = codelines[i].strip()
        if line.startswith("/**"):
            javadoc.insert(0, line)
            break
        elif line.startswith("*") or line.endswith("*/"):
            javadoc.insert(0, line)
        else:
            break  # Stop if we reach a non-comment line
    return "\n".join(javadoc)


def get_method_start_end(tree, method_node):
    start_pos = end_pos = start_line = end_line = None
    for path, node in tree:
        if start_pos is not None and method_node not in path:
            end_pos = node.position
            end_line = node.position.line if node.position is not None else None
            break
        if start_pos is None and node == method_node:
            start_pos = node.position
            start_line = node.position.line if node.position is not None else None
    return start_pos, end_pos, start_line, end_line


def get_method_text(start_pos, end_pos, start_line, end_line, last_end_line_index, codelines):
    if start_pos is None:
        return "", None, None, None
    else:
        start_line_index = start_line - 1
        end_line_index = end_line - 1 if end_pos is not None else last_end_line_index

        if last_end_line_index is not None:
            for line in codelines[(last_end_line_index + 1):(start_line_index)]:
                if "@" in line:
                    start_line_index = start_line_index - 1

        method_text = "<ST>".join(codelines[start_line_index:end_line_index])
        method_text = method_text[:method_text.rfind("}") + 1]

        if not abs(method_text.count("}") - method_text.count("{")) == 0:
            brace_diff = abs(method_text.count("}") - method_text.count("{"))
            for _ in range(brace_diff):
                method_text = method_text[:method_text.rfind("}")]

        method_lines = method_text.split("<ST>")
        method_text = "".join(method_lines)
        last_end_line_index = start_line_index + (len(method_lines) - 1)

        return method_text, (start_line_index + 1), (last_end_line_index + 1), last_end_line_index


def save_to_csv(df, filename):
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"DataFrame saved to {filename}")


# Example Usage
systems = ['prevayler', 'density-converter', 'xmlgraphics-batik', 'catena', 'h2database', 'sunflow', 'cassandra-4.0']
#systems = ['cassandra-4.0']
for system_name in systems:
    project_root = f"../data/system/{system_name}/"
    methods_index_df = build_methods_index(project_root)
    systems = {
        'prevayler': 'prevayler',
        'density-converter': 'dconverter',
        'xmlgraphics-batik': 'batik',
        'catena': 'catena',
        'h2database': 'h2',
        'sunflow': 'sunflow',
        'cassandra-4.0': 'cassandra'
    }
    updated_system_name = systems.get(system_name)
    save_path = f"./data_new/methods_index/{updated_system_name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    csv_filename = os.path.join(save_path, "methods_index.csv")
    save_to_csv(methods_index_df, csv_filename)
    print(f"Methods index saved to {csv_filename}")