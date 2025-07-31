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
import  re
import  os
import  sys
import  shutil
from    datetime    import  datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', 'src/PyThon')))
import Docy  # Assuming this module provides extract_python_functions and extract_c_functions

source_dirs = {
            'WebLog'    :   ['doc/WebLog'],
            'Devlog'    :   ['doc/Devlog'], 
            'PyThon'    :   ['src/PyThon'],
            'C'         :   ['src/C'],
            'C++'       :   ['src/CPP'],
            'CUDA'      :   ['src/CUDA'],
            }

lang_colors = {
            'PyThon': '#3572A5',
            'C': '#555555',
            'C++': '#F34B7D',
            'CUDA': "#318F8A",
            'Devlog': '#8B008B',
            'WebLog': '#076E75'
    }

output_base_dir = 'docs'



def processor(lang, content_html,file_path, base_name, processed_files,  label_to_file):
    rel_path = os.path.relpath(file_path, start=lang)
    if ("README.md" in file_path):
        html_path = os.path.join(output_base_dir, lang, os.path.split(rel_path)[0] + '.html')
        base_name = os.path.split(os.path.split(rel_path)[0])[1]

    elif(".md" in file_path):
        html_path = os.path.join(output_base_dir, lang, os.path.splitext(rel_path)[0] + '.html')
        base_name = os.path.split(os.path.split(rel_path)[0])[1]

    else:
        content_html = '\n'.join(content_html) if content_html else '<p>No functions found.</p>'
        html_path = os.path.join(output_base_dir, lang, os.path.splitext(rel_path)[0] + '.html')

    content_html, labels = Docy.process_html_for_labels(content_html)

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
        functions = Docy.extract_c_functions(file_path)
        content = [Docy.generate_c_function_html(func) for func in functions]
        return processor(lang, content,file_path, base_name, processed_files,  label_to_file)
    
    elif file_extension == ".py" and not file_path.endswith("__init__.py"):
        functions   = Docy.extract_python_functions(file_path)
        content     = [Docy.generate_python_function_html(func) for func in functions] 
        return processor(lang, content,file_path, base_name, processed_files,  label_to_file)

    elif file_extension == ".md":
        content = Docy.markdown2HTML(file_path)
        return processor(lang, content,file_path, base_name, processed_files,  label_to_file)

def build_tree(files, source_dir):
    """Build a nested dictionary for navigation structure."""
    tree = {}
    for file in files:
        rel_path = os.path.relpath(file['source_path'], source_dir)
        parts = rel_path.split(os.sep)
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = file['html_path']
    return tree

def generate_tree_html(tree, current_file_path, current_path_parts=None, indent_level=1):
    """Generate HTML for navigation tree with collapsible sections and indentation."""
    if current_path_parts is None:
        current_path_parts = []

    html = ['<ul style="margin-left: {}px;">'.format(indent_level * 20)]

    for name in sorted(tree.keys(), reverse=(tree.get('Notes') is None)):
        value = tree[name]

        # Skip README.md (handled as link for the folder itself)
        if name.lower() == 'readme.md' or name.lower() == 'readme.html':
            continue

        if isinstance(value, dict):
            # Check for README file in this folder
            readme_path = None
            for readme_name in ['README.md', 'README.html']:
                if readme_name in value:
                    readme_path = value[readme_name]
                    break

            is_open = current_path_parts and name == current_path_parts[0]
            open_attr = ' open' if is_open else ''
            sub_html = generate_tree_html(value, current_file_path, current_path_parts[1:] if is_open else [], indent_level + 1)

            if readme_path:
                rel_path = os.path.relpath(readme_path, start=os.path.dirname(current_file_path)).replace('\\', '/')
                html.append(f'<li><details{open_attr}><summary><a href="{rel_path}">{name}</a></summary>{sub_html}</details></li>')
            else:
                html.append(f'<li><details{open_attr}><summary>{name}</summary>{sub_html}</details></li>')
        else:
            rel_path = os.path.relpath(value, start=os.path.dirname(current_file_path)).replace('\\', '/')
            html.append(f'<li><a href="{rel_path}">{name}</a></li>')

    html.append('</ul>')
    return '\n'.join(html)

def create_nav_menu(processed_files, current_file_path):
    """Create navigation menu with language and devlog sections, applying distinct colors."""
    current_file = next((f for f in processed_files if f['html_path'] == current_file_path), None)
    languages = {}
    for file in processed_files:
        lang = file['lang']
        if lang not in languages:
            languages[lang] = []
        languages[lang].append(file)
    
    nav_html = ['<nav class="sidebar">', '<h2>Navigation</h2>', '<ul>'] #First line 
    for lang in source_dirs.keys():
        open_attr = ' open' if current_file  else ''

        _readmeMD = os.path.join(source_dirs[lang][0],'README.md')
        if os.path.isfile(_readmeMD):
            rel_path = os.path.relpath(_readmeMD, start=lang)
            MFile = os.path.join(output_base_dir, lang, os.path.split(rel_path)[0] + '.html')
            rel_path = os.path.relpath(MFile, start=os.path.dirname(current_file_path))
            
            nav_html.append(f'           <li class="language language-{lang.lower()}" style="border-left: 4px solid {lang_colors[lang]};"><details{open_attr}><summary><a href="{rel_path}">{lang}</a></summary>')
        else:
            nav_html.append(f'           <li class="language language-{lang.lower()}" style="border-left: 4px solid {lang_colors[lang]};"><details{open_attr}><summary>{lang}</summary>')

        
        tree = build_tree(languages[lang], source_dirs[lang][0])
        # Build and render the subtree
        if current_file and current_file['lang'] == lang:
            rel_parts = os.path.relpath(current_file['source_path'],source_dirs[lang][0]).split(os.sep)
        else:
            rel_parts = []

        # tree_html = generate_tree_html(tree, current_file_path, os.path.relpath(current_file['source_path'], source_dirs[lang][0]).split(os.sep))
        tree_html = generate_tree_html(tree, current_file_path, rel_parts)

        nav_html.append(tree_html)
        nav_html.append('</details></li>')
    nav_html.extend(['</ul>', '</nav>'])
    return '\n'.join(nav_html)

def GenerateMainPage(processed_files):
    """Generate index.html with all sections."""
    index_content = ['<h1>Project Documentation</h1>', '<div class="overview">']
    _TODOs = []
    index_content.append(f'<h2>All Files</h2><ul>')
    for file in processed_files:
        _TODOs.append(Docy.extract_tag_section(file['content_html'],'TODO'))


        rel_path = os.path.relpath(file['html_path'], start=output_base_dir)
        rel_path = rel_path.replace('\\', '/')
        # index_content.append(f'<li><a href="{rel_path}">{file["name"]}</a></li>')
        index_content.append('</ul>')
    index_content.append('</div>')

    index_content_html = '\n<h2>TODOs</h2><ul>'
    for todo in _TODOs:
        if todo:
            html = ''
            for func_name, content in todo.items():
                anchor = func_name.replace(' ', '_')  # simple anchor generation
                html += f'    <div class="todo-block" id="{anchor}">\n'
                html += f'        <h2><a href="#{anchor}" class="function-link">{func_name}</a></h2>\n'
                html += f'        <ul>\n'
                for line in content.strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('- [ ]'):
                        html += f'            <li><input type="checkbox" class="checkbox" disabled>{line[5:].strip()}</li>\n'
                    else:
                        html += f'            <li>{line}</li>\n'
                html += f'        </ul>\n'
                html += f'    </div>\n'



            index_content_html += f'<li>{html}</li>'
    index_content_html += '</ul>'

    index_content_html +='\n'.join(index_content)

    with open('doc/template.html', 'r') as f:
        template = f.read()

    file = {}
    file['html_path'] =  os.path.join(output_base_dir, 'index.html')
    
    output = template.replace('<!-- TITLE -->', 'Project Documentation')
    output = output.replace('<!-- styles -->', Docy.css_styles(file))
    output = output.replace('<!-- NAVIGATION -->',  create_nav_menu(processed_files, file['html_path']))
    output = output.replace('<!-- CONTENT -->', index_content_html)
    output = output.replace('<!-- GENERATION_DATE -->', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    output = output.replace('<!-- Adress -->', '<a href="https://github.com/YassinRiyazi/Main">https://github.com/YassinRiyazi/Main</a>')

    with open(os.path.join(output_base_dir, 'index.html'), 'w') as f:
        f.write(output)

def main(source_dirs):
    if os.path.isdir(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    shutil.copytree(os.path.join('doc', 'styles'), os.path.join(output_base_dir, 'styles'), dirs_exist_ok=True)

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
                        process_decider(lang, file_path, filename, file_extension, processed_files, label_to_file, output_base_dir)

    # file_name_to_html_path = Docy.fileNameExtractor(processed_files, source_dirs)
    file_name_to_html_path = Docy.fileNameExtractor_Langless(processed_files, source_dirs)

    # Generate HTML for all files with file name hyperlinks
    _tempWebAdress = "https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/"
    for file in processed_files:
        with open('doc/template.html', 'r') as f:
            template = f.read()

        if file['type'] == 'source':
            title = f"{file['name']} "
        elif file['type'] == 'notes':
            title = f"{file['lang']} Notes"
        elif file['type'] == 'Devlog':
            title = f""
        elif file['type'] == 'WebLog':
            title = f""

        # content_html = Docy.replace_file_names_in_html(file['content_html'], file_name_to_html_path.get(file['lang'], {}), file['html_path'])
        content_html = Docy.replace_file_names_in_html(file['content_html'], file_name_to_html_path, file['html_path'])
        output = template.replace('<!-- TITLE -->', title)
        output = output.replace('<!-- styles -->', Docy.css_styles(file))
        output = output.replace('<!-- NAVIGATION -->', create_nav_menu(processed_files, file['html_path']))
        output = output.replace('<!-- CONTENT -->', content_html)#content_html
        output = output.replace('<!-- GENERATION_DATE -->', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        output = output.replace('<!-- Adress -->', f'<a href="{_tempWebAdress}{file["source_path"]}">{_tempWebAdress}{file["source_path"]}</a>')
        with open(file['html_path'], 'w') as f:
            f.write(output)

    
    Docy.process_html_for_labels_replace(processed_files, label_to_file)

    GenerateMainPage(processed_files)
            

if __name__ == "__main__":
    main(source_dirs)


    """
    Unified file processing instead having different branch for each file.
    """