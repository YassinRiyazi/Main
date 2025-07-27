"""
    * Each folder should have an entry md file to desctibe its content and functions
    * A loop over All valid first level folders


    Navigation panel: 
        Src [1st level]
            Python
                Notes [2nd level]
                    ...
                projects
                    ...
                    
            C
                Notes 
                    ...
                projects
                    ...

        Weblog [1st level]
            Folder 1 [2nd level]

        Devlog [1st level]

"""
import re
import os
import maker  # Assuming this module provides extract_python_functions and extract_c_functions
import markdown
from datetime import datetime

LangLists = {
            'PyThon'    :   'src/PyThon',
            'C'         :   'src/C',
            'C++'       :   'src/CPP',
            'CUDA'      :   'src/CUDA',}

DevLog  = { 'DevLog'    :   'doc/Devlog'}
WebLog  = { 'WebLog'    :   'doc/WebLog'}

source_dirs = {
            'src'       : LangLists,
            'DevLog'    : DevLog,
            'WebLog'    : WebLog}

output_base_dir = 'docs'

def markdown2HTML(file_path):
    """Convert markdown to HTML."""
    with open(file_path, 'r') as f:
        markdown_content = f.read()
    html_content = markdown.markdown(markdown_content)
    return html_content

def process_html_for_labels(html_content):
    """Process HTML content to replace \label tags with anchors and collect labels."""
    labels = []
    def replace_label(match):
        label = match.group(1)
        labels.append(label)
        return f'<p> <span class="keyword">{label.split(r"\\Label: ")[0]}</span></p><span id="{label}"></span>'
    processed_html = re.sub(r'\\Label:\s*(\w+)', replace_label, html_content)
    return processed_html, labels

def processor(_Type, lang, content,file_path, base_name, processed_files,  label_to_file):
    content_html = '\n'.join(content) if content else '<p>No functions found.</p>'
    content_html, labels = process_html_for_labels(content_html)

    rel_path = os.path.relpath(file_path, start=source_dirs[_Type][lang])
    html_path = os.path.join(output_base_dir, lang, os.path.splitext(rel_path)[0] + '.html')
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    processed_files.append({
                            'content_html': content_html,
                            'lang': lang,
                            'source_path': file_path,
                            'html_path': html_path,
                            'name': base_name,
                            'type': 'source',
                            'labels': labels
                        })
    for label in labels:
        if label in label_to_file:
            print(f"Warning: duplicate label '{label}' in {html_path} and {label_to_file[label]}")
        label_to_file[label] = html_path
    return processed_files, label_to_file

def process_decider(_Type, lang,file_path, base_name, file_extension, processed_files, label_to_file, output_base_dir):
    if file_extension == ".c":
        functions = maker.extract_c_functions(file_path)
        content = [maker.generate_c_function_html(func) for func in functions]
        return processor(_Type, lang, content,file_path, base_name, processed_files,  label_to_file)
    
    elif file_extension == ".py" and not file_path.endswith("__init__.py"):
        functions   = maker.extract_python_functions(file_path)
        content     = [maker.generate_python_function_html(func) for func in functions] 
        return processor(_Type, lang, content,file_path, base_name, processed_files,  label_to_file)

    elif file_extension == ".md":
        content = markdown2HTML(file_path)
        return processor(_Type, lang, content,file_path, base_name, processed_files,  label_to_file)


def walkthrough():
    # Process source files
    processed_files = []
    label_to_file = {}
    for _Type, dicks in source_dirs.items():
        for lang, dir_path in dicks.items():
                if os.path.exists(dir_path):
                    for root, _, files in os.walk(dir_path):
                        for file in files:
                            filename, file_extension = os.path.splitext(file)
                            file_path = os.path.join(root, file)
                            # processed_files, label_to_file = 
                            process_decider(_Type, lang, file_path, filename, file_extension, processed_files, label_to_file, output_base_dir)
                            print(filename, file_extension, lang, _Type)

if __name__ == "__main__":
    walkthrough()


    """
    Unified file processing instead having different branch for each file.
    """