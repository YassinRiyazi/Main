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
import shutil
import markdown
from datetime import datetime

source_dirs = {
            'src'       : ['src/PyThon','src/C','src/CPP','src/CUDA'],#LangLists,
            'Devlog'    : ['doc/Devlog'], #DevLog,
            'WebLog'    : ['doc/WebLog'],} #WebLog}

output_base_dir = 'docs2'

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

def processor(lang, content,file_path, base_name, processed_files,  label_to_file):
    content_html = '\n'.join(content) if content else '<p>No functions found.</p>'
    content_html, labels = process_html_for_labels(content_html)

    rel_path = os.path.relpath(file_path, start=lang)
    if ("README.md" in file_path):
        html_path = os.path.join(output_base_dir, lang, os.path.split(rel_path)[0] + '.html')
    else:
        html_path = os.path.join(output_base_dir, lang, os.path.splitext(rel_path)[0] + '.html')
    html_path = os.path.normpath(html_path)
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    processed_files.append({
                            'content_html': content_html,
                            'lang': lang,
                            'source_path': file_path,
                            'html_path': html_path,
                            'name': base_name,
                            'type': 'source', #TODO
                            'labels': labels
                        })
    for label in labels:
        if label in label_to_file:
            print(f"Warning: duplicate label '{label}' in {html_path} and {label_to_file[label]}")
        label_to_file[label] = html_path
    return processed_files, label_to_file

def process_decider(lang,file_path, base_name, file_extension, processed_files, label_to_file, output_base_dir):
    if file_extension == ".c":
        functions = maker.extract_c_functions(file_path)
        content = [maker.generate_c_function_html(func) for func in functions]
        return processor(lang, content,file_path, base_name, processed_files,  label_to_file)
    
    elif file_extension == ".py" and not file_path.endswith("__init__.py"):
        functions   = maker.extract_python_functions(file_path)
        content     = [maker.generate_python_function_html(func) for func in functions] 
        return processor(lang, content,file_path, base_name, processed_files,  label_to_file)

    elif file_extension == ".md":
        content = markdown2HTML(file_path)
        return processor(lang, content,file_path, base_name, processed_files,  label_to_file)


def main():
    if os.path.isdir(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    # Process source files
    processed_files = []
    label_to_file = {}
    file_name_to_html_path = {}
    for lang, lists in source_dirs.items():
        for dir_path in lists:
            if os.path.exists(dir_path):
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        filename, file_extension = os.path.splitext(file)
                        file_path = os.path.join(root, file)
                        # processed_files, label_to_file = 
                        process_decider(lang, file_path, filename, file_extension, processed_files, label_to_file, output_base_dir)

                        if processed_files[-1]['type'] == 'source':
                            base_name = os.path.splitext(os.path.basename(processed_files[-1]['source_path']))[0]
                            if base_name not in file_name_to_html_path[lang]:
                                file_name_to_html_path[lang][base_name] = processed_files[-1]['html_path']
                            else:
                                print(f"Warning: duplicate file name '{base_name}' in language {lang}")
                        elif processed_files[-1]['type'] == 'notes':
                            file_name_to_html_path[lang]['notes'] = processed_files[-1]['html_path']
                        elif processed_files[-1]['type'] in ['Devlog', 'WebLog']:
                            file_name_to_html_path[lang][processed_files[-1]['name']] = processed_files[-1]['html_path']
    # for i in processed_files:
    #     print(i["source_path"])
    # Build file name to HTML path mapping

            

if __name__ == "__main__":
    main()


    """
    Unified file processing instead having different branch for each file.
    """