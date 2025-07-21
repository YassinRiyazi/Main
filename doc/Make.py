# import os
# import shutil
# from datetime import datetime
# import markdown
# import maker  # Assuming this module provides extract_python_functions and extract_c_functions

# # Define source directories for each language and notes
# source_dirs = {
#     'Python': 'src/PyThon',
#     'C': 'src/C',
#     'C++': 'src/CPP',
#     'CUDA': 'src/CUDA'
# }
# notes_dir = 'doc/notes'
# output_base_dir = 'docs'

# def get_language_from_path(html_path, output_base_dir):
#     """Determine the language from the HTML path."""
#     rel_path = os.path.relpath(html_path, output_base_dir)
#     parts = rel_path.split(os.sep)
#     if len(parts) >= 2 and parts[0] in ['Python', 'C', 'C++', 'CUDA']:
#         return parts[0]
#     return None

# def generate_content_html(file_path):
#     """Generate HTML content for functions in the given source file."""
#     if file_path.endswith('.py') and not file_path.endswith("__init__.py"):
#         functions = maker.extract_python_functions(file_path)
#         content = [maker.generate_python_function_html(func) for func in functions]
#         lang = 'Python'
#     elif file_path.endswith('.c'):
#         functions = maker.extract_c_functions(file_path)
#         content = [maker.generate_c_function_html(func) for func in functions]
#         lang = 'C'
#     else:
#         return None, None
#     return '\n'.join(content) if content else '<p>No functions found.</p>', lang

# def generate_notes_html(file_path, lang):
#     """Convert markdown notes to HTML."""
#     if not os.path.exists(file_path):
#         return f'<p>No {lang} notes available.</p>', lang
#     with open(file_path, 'r') as f:
#         markdown_content = f.read()
#     html_content = markdown.markdown(markdown_content)
#     return html_content, lang

# def build_tree(files, source_dir):
#     """Build a nested dictionary representing the directory structure."""
#     tree = {}
#     for file in files:
#         if file['type'] == 'notes':
#             tree['Notes'] = file['html_path']
#         else:
#             rel_path = os.path.relpath(file['source_path'], source_dir)
#             parts = rel_path.split(os.sep)
#             current = tree
#             for part in parts[:-1]:
#                 if part not in current:
#                     current[part] = {}
#                 current = current[part]
#             current[parts[-1]] = file['html_path']
#     return tree

# def generate_tree_html(tree, current_file_path, current_path_parts=None):
#     """Recursively generate HTML for the directory tree with collapsible sections."""
#     if current_path_parts is None:
#         current_path_parts = []
#     html = ['<ul>']
#     for name in sorted(tree.keys()):
#         value = tree[name]
#         if isinstance(value, dict):
#             # Directory
#             is_open = current_path_parts and name == current_path_parts[0]
#             open_attr = ' open' if is_open else ''
#             sub_html = generate_tree_html(value, current_file_path, current_path_parts[1:] if is_open else [])
#             html.append(f'<li><details{open_attr}><summary>{name}</summary>{sub_html}</details></li>')
#         else:
#             # File
#             rel_path = os.path.relpath(value, start=os.path.dirname(current_file_path))
#             rel_path = rel_path.replace('\\', '/')
#             html.append(f'<li><a href="{rel_path}">{name}</a></li>')
#     html.append('</ul>')
#     return '\n'.join(html)

# def create_nav_menu(processed_files, current_file_path):
#     """Create the navigation menu with collapsible sections."""
#     current_file = next((f for f in processed_files if f['html_path'] == current_file_path), None)
#     languages = {}
#     for file in processed_files:
#         lang = file['lang']
#         if lang not in languages:
#             languages[lang] = []
#         languages[lang].append(file)
    
#     nav_html = ['<nav class="sidebar">', '<h2>Navigation</h2>', '<ul>']
#     for lang in ['Python', 'C', 'C++', 'CUDA']:
#         if lang in languages and languages[lang]:
#             open_attr = ' open' if current_file and current_file['lang'] == lang else ''
#             if current_file and current_file['lang'] == lang:
#                 if current_file['type'] == 'notes':
#                     current_path_parts = ['Notes']
#                 else:
#                     rel_path = os.path.relpath(current_file['source_path'], source_dirs[lang])
#                     current_path_parts = rel_path.split(os.sep)
#             else:
#                 current_path_parts = []
#             nav_html.append(f'<li class="language"><details{open_attr}><summary>{lang}</summary>')
#             tree = build_tree(languages[lang], source_dirs[lang])
#             tree_html = generate_tree_html(tree, current_file_path, current_path_parts)
#             nav_html.append(tree_html)
#             nav_html.append('</details></li>')
#     nav_html.extend(['</ul>', '</nav>'])
#     return '\n'.join(nav_html)

# def GenerateMainPage(LangLists, processed_files):
#     """Generate index.html."""
#     index_content = ['<h1>Project Documentation</h1>', '<div class="overview">']
#     for lang in LangLists:
#         lang_files = [f for f in processed_files if f['lang'] == lang]
#         if lang_files:
#             index_content.append(f'<h2>{lang} Files</h2><ul>')
#             for file in lang_files:
#                 rel_path = os.path.relpath(file['html_path'], start=output_base_dir)
#                 rel_path = rel_path.replace('\\', '/')
#                 index_content.append(f'<li><a href="{rel_path}">{file["name"]}</a></li>')
#             index_content.append('</ul>')
#     index_content.append('</div>')
#     index_content_html = '\n'.join(index_content)

#     with open('doc/template.html', 'r') as f:
#         template = f.read()

#     nav_menu = create_nav_menu(processed_files, os.path.join(output_base_dir, 'index.html'))
#     output = template.replace('<!-- TITLE -->', 'Project Documentation')
#     output = output.replace('<!-- NAVIGATION -->', nav_menu)
#     output = output.replace('<!-- CONTENT -->', index_content_html)
#     output = output.replace('<!-- GENERATION_DATE -->', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#     output = output.replace('<!-- Adress -->', '<a href="https://github.com/YassinRiyazi/Main">https://github.com/YassinRiyazi/Main</a>')

#     with open(os.path.join(output_base_dir, 'index.html'), 'w') as f:
#         f.write(output)

# def main():
#     if os.path.isdir(output_base_dir):
#         shutil.rmtree(output_base_dir)
#     os.makedirs(output_base_dir, exist_ok=True)

#     LangLists = ['Python', 'C', 'C++', 'CUDA']
#     processed_files = []
#     # Process source files
#     for lang, dir_path in source_dirs.items():
#         if os.path.exists(dir_path):
#             for root, _, files in os.walk(dir_path):
#                 for file in files:
#                     if file.endswith(('.py', '.c', '.cu', '.h', '.cpp', '.hpp')):
#                         file_path = os.path.join(root, file)
#                         content_html, file_lang = generate_content_html(file_path)
#                         if content_html:
#                             base_name = os.path.splitext(file)[0]
#                             rel_path = os.path.relpath(file_path, start=source_dirs[lang])
#                             html_path = os.path.join(output_base_dir, lang, os.path.splitext(rel_path)[0] + '.html')
#                             os.makedirs(os.path.dirname(html_path), exist_ok=True)
#                             processed_files.append({
#                                 'content_html': content_html,
#                                 'lang': file_lang,
#                                 'source_path': file_path,
#                                 'html_path': html_path,
#                                 'name': base_name,
#                                 'type': 'source'
#                             })
    
#     # Process notes files
#     for lang in LangLists:
#         notes_file = os.path.join(notes_dir, f'notes_{lang.lower()}.md')
#         content_html, file_lang = generate_notes_html(notes_file, lang)
#         html_path = os.path.join(output_base_dir, lang, 'notes.html')
#         os.makedirs(os.path.dirname(html_path), exist_ok=True)
#         processed_files.append({
#             'content_html': content_html,
#             'lang': file_lang,
#             'source_path': notes_file,
#             'html_path': html_path,
#             'name': f'{lang} Notes',
#             'type': 'notes'
#         })
    
#     _tempWebAdress = "https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/"
#     # Generate full HTML for each file
#     for file in processed_files:
#         with open('doc/template.html', 'r') as f:
#             template = f.read()
#         nav_menu = create_nav_menu(processed_files, file['html_path'])
#         output = template.replace('<!-- TITLE -->', f"{file['name']} Documentation")
#         output = output.replace('<!-- NAVIGATION -->', nav_menu)
#         output = output.replace('<!-- CONTENT -->', file['content_html'])
#         output = output.replace('<!-- GENERATION_DATE -->', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#         output = output.replace('<!-- Adress -->', f'<a href="{_tempWebAdress}{file["source_path"]}">{_tempWebAdress}{file["source_path"]}</a>')

#         with open(file['html_path'], 'w') as f:
#             f.write(output)
    
#     GenerateMainPage(LangLists, processed_files)

# if __name__ == "__main__":
#     main()

# import os
# import shutil
# from datetime import datetime
# import markdown
# import maker  # Assuming this module provides extract_python_functions and extract_c_functions

# # Define source directories for each language, notes, and devlog
# source_dirs = {
#     'Python': 'src/PyThon',
#     'C': 'src/C',
#     'C++': 'src/CPP',
#     'CUDA': 'src/CUDA',
#     'Devlog': 'devlog'
# }
# notes_dir = 'doc/notes'
# output_base_dir = 'docs'

# def get_language_from_path(html_path, output_base_dir):
#     """Determine the language or section from the HTML path."""
#     rel_path = os.path.relpath(html_path, output_base_dir)
#     parts = rel_path.split(os.sep)
#     if len(parts) >= 2 and parts[0] in ['Python', 'C', 'C++', 'CUDA', 'Devlog']:
#         return parts[0]
#     return None

# def generate_content_html(file_path):
#     """Generate HTML content for functions in the given source file."""
#     if file_path.endswith('.py') and not file_path.endswith("__init__.py"):
#         functions = maker.extract_python_functions(file_path)
#         content = [maker.generate_python_function_html(func) for func in functions]
#         lang = 'Python'
#     elif file_path.endswith('.c'):
#         functions = maker.extract_c_functions(file_path)
#         content = [maker.generate_c_function_html(func) for func in functions]
#         lang = 'C'
#     elif file_path.endswith('.md'):
#         with open(file_path, 'r') as f:
#             markdown_content = f.read()
#         content = [markdown.markdown(markdown_content)]
#         lang = 'Devlog'
#     else:
#         return None, None
#     return '\n'.join(content) if content else '<p>No content found.</p>', lang

# def generate_notes_html(file_path, lang):
#     """Convert markdown notes to HTML."""
#     if not os.path.exists(file_path):
#         return f'<p>No {lang} notes available.</p>', lang
#     with open(file_path, 'r') as f:
#         markdown_content = f.read()
#     html_content = markdown.markdown(markdown_content)
#     return html_content, lang

# def build_tree(files, source_dir):
#     """Build a nested dictionary representing the directory structure."""
#     tree = {}
#     for file in files:
#         if file['type'] == 'notes':
#             tree['Notes'] = file['html_path']
#         elif file['type'] == 'devlog':
#             base_name = file['name']
#             tree[base_name] = file['html_path']
#         else:
#             rel_path = os.path.relpath(file['source_path'], source_dir)
#             parts = rel_path.split(os.sep)
#             current = tree
#             for part in parts[:-1]:
#                 if part not in current:
#                     current[part] = {}
#                 current = current[part]
#             current[parts[-1]] = file['html_path']
#     return tree

# def generate_tree_html(tree, current_file_path, current_path_parts=None):
#     """Recursively generate HTML for the directory tree with collapsible sections."""
#     if current_path_parts is None:
#         current_path_parts = []
#     html = ['<ul>']
#     for name in sorted(tree.keys()):
#         value = tree[name]
#         if isinstance(value, dict):
#             # Directory
#             is_open = current_path_parts and name == current_path_parts[0]
#             open_attr = ' open' if is_open else ''
#             sub_html = generate_tree_html(value, current_file_path, current_path_parts[1:] if is_open else [])
#             html.append(f'<li><details{open_attr}><summary>{name}</summary>{sub_html}</details></li>')
#         else:
#             # File
#             rel_path = os.path.relpath(value, start=os.path.dirname(current_file_path))
#             rel_path = rel_path.replace('\\', '/')
#             html.append(f'<li><a href="{rel_path}">{name}</a></li>')
#     html.append('</ul>')
#     return '\n'.join(html)

# def create_nav_menu(processed_files, current_file_path):
#     """Create the navigation menu with collapsible sections."""
#     current_file = next((f for f in processed_files if f['html_path'] == current_file_path), None)
#     languages = {}
#     for file in processed_files:
#         lang = file['lang']
#         if lang not in languages:
#             languages[lang] = []
#         languages[lang].append(file)
    
#     nav_html = ['<nav class="sidebar">', '<h2>Navigation</h2>', '<ul>']
#     for lang in ['Python', 'C', 'C++', 'CUDA', 'Devlog']:
#         if lang in languages and languages[lang]:
#             open_attr = ' open' if current_file and current_file['lang'] == lang else ''
#             if current_file and current_file['lang'] == lang:
#                 if current_file['type'] == 'notes':
#                     current_path_parts = ['Notes']
#                 elif current_file['type'] == 'devlog':
#                     current_path_parts = [current_file['name']]
#                 else:
#                     rel_path = os.path.relpath(current_file['source_path'], source_dirs[lang])
#                     current_path_parts = rel_path.split(os.sep)
#             else:
#                 current_path_parts = []
#             nav_html.append(f'<li class="language"><details{open_attr}><summary>{lang}</summary>')
#             tree = build_tree(languages[lang], source_dirs[lang])
#             tree_html = generate_tree_html(tree, current_file_path, current_path_parts)
#             nav_html.append(tree_html)
#             nav_html.append('</details></li>')
#     nav_html.extend(['</ul>', '</nav>'])
#     return '\n'.join(nav_html)

# def GenerateMainPage(LangLists, processed_files):
#     """Generate index.html."""
#     index_content = ['<h1>Project Documentation</h1>', '<div class="overview">']
#     for lang in LangLists:
#         lang_files = [f for f in processed_files if f['lang'] == lang]
#         if lang_files:
#             index_content.append(f'<h2>{lang} Files</h2><ul>')
#             for file in lang_files:
#                 rel_path = os.path.relpath(file['html_path'], start=output_base_dir)
#                 rel_path = rel_path.replace('\\', '/')
#                 index_content.append(f'<li><a href="{rel_path}">{file["name"]}</a></li>')
#             index_content.append('</ul>')
#     index_content.append('</div>')
#     index_content_html = '\n'.join(index_content)

#     with open('doc/template.html', 'r') as f:
#         template = f.read()

#     nav_menu = create_nav_menu(processed_files, os.path.join(output_base_dir, 'index.html'))
#     output = template.replace('<!-- TITLE -->', 'Project Documentation')
#     output = output.replace('<!-- NAVIGATION -->', nav_menu)
#     output = output.replace('<!-- CONTENT -->', index_content_html)
#     output = output.replace('<!-- GENERATION_DATE -->', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#     output = output.replace('<!-- Adress -->', '<a href="https://github.com/YassinRiyazi/Main">https://github.com/YassinRiyazi/Main</a>')

#     with open(os.path.join(output_base_dir, 'index.html'), 'w') as f:
#         f.write(output)

# def main():
#     if os.path.isdir(output_base_dir):
#         shutil.rmtree(output_base_dir)
#     os.makedirs(output_base_dir, exist_ok=True)

#     LangLists = ['Python', 'C', 'C++', 'CUDA', 'Devlog']
#     processed_files = []
#     # Process source files
#     for lang, dir_path in source_dirs.items():
#         if os.path.exists(dir_path):
#             for root, _, files in os.walk(dir_path):
#                 for file in files:
#                     if (lang != 'Devlog' and file.endswith(('.py', '.c', '.cu', '.h', '.cpp', '.hpp'))) or \
#                        (lang == 'Devlog' and file.endswith('.md')):
#                         file_path = os.path.join(root, file)
#                         content_html, file_lang = generate_content_html(file_path)
#                         if content_html:
#                             base_name = os.path.splitext(file)[0]
#                             rel_path = os.path.relpath(file_path, start=source_dirs[lang])
#                             html_path = os.path.join(output_base_dir, lang, os.path.splitext(rel_path)[0] + '.html')
#                             os.makedirs(os.path.dirname(html_path), exist_ok=True)
#                             processed_files.append({
#                                 'content_html': content_html,
#                                 'lang': file_lang,
#                                 'source_path': file_path,
#                                 'html_path': html_path,
#                                 'name': base_name,
#                                 'type': 'source' if lang != 'Devlog' else 'devlog'
#                             })
    
#     # Process notes files
#     for lang in ['Python', 'C', 'C++', 'CUDA']:
#         notes_file = os.path.join(notes_dir, f'notes_{lang.lower()}.md')
#         content_html, file_lang = generate_notes_html(notes_file, lang)
#         html_path = os.path.join(output_base_dir, lang, 'notes.html')
#         os.makedirs(os.path.dirname(html_path), exist_ok=True)
#         processed_files.append({
#             'content_html': content_html,
#             'lang': file_lang,
#             'source_path': notes_file,
#             'html_path': html_path,
#             'name': f'{lang} Notes',
#             'type': 'notes'
#         })
    
#     _tempWebAdress = "https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/"
#     # Generate full HTML for each file
#     for file in processed_files:
#         with open('doc/template.html', 'r') as f:
#             template = f.read()
#         nav_menu = create_nav_menu(processed_files, file['html_path'])
#         output = template.replace('<!-- TITLE -->', f"{file['name']} Documentation")
#         output = output.replace('<!-- NAVIGATION -->', nav_menu)
#         output = output.replace('<!-- CONTENT -->', file['content_html'])
#         output = output.replace('<!-- GENERATION_DATE -->', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#         output = output.replace('<!-- Adress -->', f'<a href="{_tempWebAdress}{file["source_path"]}">{_tempWebAdress}{file["source_path"]}</a>')

#         with open(file['html_path'], 'w') as f:
#             f.write(output)
    
#     GenerateMainPage(LangLists, processed_files)

# if __name__ == "__main__":
#     main()


import os
import shutil
from datetime import datetime
import markdown
import maker  # Assuming this module provides extract_python_functions and extract_c_functions

# Define source directories for each language, notes, and devlog
source_dirs = {
    'Python': 'src/PyThon',
    'C': 'src/C',
    'C++': 'src/CPP',
    'CUDA': 'src/CUDA'
}
notes_dir = 'doc/notes'
devlog_dir = 'doc/Devlog'
output_base_dir = 'docs'

def get_language_from_path(html_path, output_base_dir):
    """Determine the language from the HTML path."""
    rel_path = os.path.relpath(html_path, output_base_dir)
    parts = rel_path.split(os.sep)
    if len(parts) >= 2 and parts[0] in ['Python', 'C', 'C++', 'CUDA']:
        return parts[0]
    return None

def generate_content_html(file_path):
    """Generate HTML content for functions in the given source file."""
    if file_path.endswith('.py') and not file_path.endswith("__init__.py"):
        functions = maker.extract_python_functions(file_path)
        content = [maker.generate_python_function_html(func) for func in functions]
        lang = 'Python'
    elif file_path.endswith('.c'):
        functions = maker.extract_c_functions(file_path)
        content = [maker.generate_c_function_html(func) for func in functions]
        lang = 'C'
    else:
        return None, None
    return '\n'.join(content) if content else '<p>No functions found.</p>', lang

def generate_notes_html(file_path, lang):
    """Convert markdown notes to HTML."""
    if not os.path.exists(file_path):
        return f'<p>No {lang} notes available.</p>', lang
    with open(file_path, 'r') as f:
        markdown_content = f.read()
    html_content = markdown.markdown(markdown_content)
    return html_content, lang

def generate_devlog_html(file_path):
    """Convert markdown devlog to HTML."""
    with open(file_path, 'r') as f:
        markdown_content = f.read()
    html_content = markdown.markdown(markdown_content)
    return html_content

def build_tree(files, source_dir):
    """Build a nested dictionary representing the directory structure."""
    tree = {}
    for file in files:
        if file['type'] == 'notes':
            tree['Notes'] = file['html_path']
        else:
            rel_path = os.path.relpath(file['source_path'], source_dir)
            parts = rel_path.split(os.sep)
            current = tree
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = file['html_path']
    return tree

def generate_tree_html(tree, current_file_path, current_path_parts=None):
    """Recursively generate HTML for the directory tree with collapsible sections."""
    if current_path_parts is None:
        current_path_parts = []
    html = ['<ul>']
    for name in sorted(tree.keys()):
        value = tree[name]
        if isinstance(value, dict):
            is_open = current_path_parts and name == current_path_parts[0]
            open_attr = ' open' if is_open else ''
            sub_html = generate_tree_html(value, current_file_path, current_path_parts[1:] if is_open else [])
            html.append(f'<li><details{open_attr}><summary>{name}</summary>{sub_html}</details></li>')
        else:
            rel_path = os.path.relpath(value, start=os.path.dirname(current_file_path))
            rel_path = rel_path.replace('\\', '/')
            html.append(f'<li><a href="{rel_path}">{name}</a></li>')
    html.append('</ul>')
    return '\n'.join(html)

def create_nav_menu(processed_files, current_file_path):
    """Create the navigation menu with collapsible sections."""
    current_file = next((f for f in processed_files if f['html_path'] == current_file_path), None)
    languages = {}
    for file in processed_files:
        if file['type'] != 'devlog':  # Exclude devlogs from sidebar navigation
            lang = file['lang']
            if lang not in languages:
                languages[lang] = []
            languages[lang].append(file)
    
    nav_html = ['<nav class="sidebar">', '<h2>Navigation</h2>', '<ul>']
    for lang in ['Python', 'C', 'C++', 'CUDA']:
        if lang in languages and languages[lang]:
            open_attr = ' open' if current_file and current_file['lang'] == lang else ''
            if current_file and current_file['lang'] == lang:
                if current_file['type'] == 'notes':
                    current_path_parts = ['Notes']
                else:
                    rel_path = os.path.relpath(current_file['source_path'], source_dirs[lang])
                    current_path_parts = rel_path.split(os.sep)
            else:
                current_path_parts = []
            nav_html.append(f'<li class="language"><details{open_attr}><summary>{lang}</summary>')
            tree = build_tree(languages[lang], source_dirs[lang])
            tree_html = generate_tree_html(tree, current_file_path, current_path_parts)
            nav_html.append(tree_html)
            nav_html.append('</details></li>')
    nav_html.extend(['</ul>', '</nav>'])
    return '\n'.join(nav_html)

def generate_dropdown_menu(devlog_files, output_base_dir):
    """Generate the drop-down menu HTML for the header."""
    menu_html = ['<nav class="dropdown-menu">', '<ul>', '<li><a href="/index.html">Main Page</a></li>', '<li><details><summary>Dev Logs</summary><ul>']
    for file in devlog_files:
        rel_path = os.path.relpath(file['html_path'], output_base_dir).replace('\\', '/')
        menu_html.append(f'<li><a href="/{rel_path}">{file["name"]}</a></li>')
    menu_html.extend(['</ul></details></li>', '</ul>', '</nav>'])
    return '\n'.join(menu_html)

def GenerateMainPage(LangLists, processed_files, dropdown_menu_html):
    """Generate index.html with the drop-down menu."""
    index_content = ['<h1>Project Documentation</h1>', '<div class="overview">']
    for lang in LangLists:
        lang_files = [f for f in processed_files if f['lang'] == lang and f['type'] != 'devlog']
        if lang_files:
            index_content.append(f'<h2>{lang} Files</h2><ul>')
            for file in lang_files:
                rel_path = os.path.relpath(file['html_path'], start=output_base_dir)
                rel_path = rel_path.replace('\\', '/')
                index_content.append(f'<li><a href="{rel_path}">{file["name"]}</a></li>')
            index_content.append('</ul>')
    index_content.append('</div>')
    index_content_html = '\n'.join(index_content)

    with open('doc/template.html', 'r') as f:
        template = f.read()

    nav_menu = create_nav_menu(processed_files, os.path.join(output_base_dir, 'index.html'))
    output = template.replace('<!-- TITLE -->', 'Project Documentation')
    output = output.replace('<!-- NAVIGATION -->', nav_menu)
    output = output.replace('<!-- CONTENT -->', index_content_html)
    output = output.replace('<!-- GENERATION_DATE -->', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    output = output.replace('<!-- Adress -->', '<a href="https://github.com/YassinRiyazi/Main">https://github.com/YassinRiyazi/Main</a>')
    output = output.replace('<!-- DROPDOWN_MENU -->', dropdown_menu_html)

    with open(os.path.join(output_base_dir, 'index.html'), 'w') as f:
        f.write(output)

def main():
    if os.path.isdir(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    LangLists = ['Python', 'C', 'C++', 'CUDA']
    processed_files = []

    # Process source files
    for lang, dir_path in source_dirs.items():
        if os.path.exists(dir_path):
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(('.py', '.c', '.cu', '.h', '.cpp', '.hpp')):
                        file_path = os.path.join(root, file)
                        content_html, file_lang = generate_content_html(file_path)
                        if content_html:
                            base_name = os.path.splitext(file)[0]
                            rel_path = os.path.relpath(file_path, start=source_dirs[lang])
                            html_path = os.path.join(output_base_dir, lang, os.path.splitext(rel_path)[0] + '.html')
                            os.makedirs(os.path.dirname(html_path), exist_ok=True)
                            processed_files.append({
                                'content_html': content_html,
                                'lang': file_lang,
                                'source_path': file_path,
                                'html_path': html_path,
                                'name': base_name,
                                'type': 'source'
                            })

    # Process notes files
    for lang in LangLists:
        notes_file = os.path.join(notes_dir, f'notes_{lang.lower()}.md')
        content_html, file_lang = generate_notes_html(notes_file, lang)
        html_path = os.path.join(output_base_dir, lang, 'notes.html')
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        processed_files.append({
            'content_html': content_html,
            'lang': file_lang,
            'source_path': notes_file,
            'html_path': html_path,
            'name': f'{lang} Notes',
            'type': 'notes'
        })

    # Process devlog files
    if os.path.exists(devlog_dir):
        for root, dirs, files in os.walk(devlog_dir):
            if 'README.md' in files:
                readme_path = os.path.join(root, 'README.md')
                rel_path = os.path.relpath(root, devlog_dir)
                if rel_path == '.':
                    continue  # Skip README.md directly in devlog/
                subdir_name = os.path.basename(root)
                html_path = os.path.join(output_base_dir, 'devlog', rel_path + '.html')
                os.makedirs(os.path.dirname(html_path), exist_ok=True)
                html_content = generate_devlog_html(readme_path)
                processed_files.append({
                    'content_html': html_content,
                    'lang': None,
                    'source_path': readme_path,
                    'html_path': html_path,
                    'name': subdir_name,
                    'type': 'devlog'
                })

    # Generate drop-down menu
    devlog_files = [f for f in processed_files if f['type'] == 'devlog']
    dropdown_menu_html = generate_dropdown_menu(devlog_files, output_base_dir)

    _tempWebAdress = "https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/"

    # Generate full HTML for each file
    for file in processed_files:
        with open('doc/template.html', 'r') as f:
            template = f.read()
        nav_menu = create_nav_menu(processed_files, file['html_path'])
        if file['type'] == 'source':
            title = f"{file['name']} Documentation"
        elif file['type'] == 'notes':
            title = f"{file['lang']} Notes"
        elif file['type'] == 'devlog':
            title = f"Dev Log: {file['name']}"
        output = template.replace('<!-- TITLE -->', title)
        output = output.replace('<!-- NAVIGATION -->', nav_menu)
        output = output.replace('<!-- CONTENT -->', file['content_html'])
        output = output.replace('<!-- GENERATION_DATE -->', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        output = output.replace('<!-- Adress -->', f'<a href="{_tempWebAdress}{file["source_path"]}">{_tempWebAdress}{file["source_path"]}</a>')
        output = output.replace('<!-- DROPDOWN_MENU -->', dropdown_menu_html)
        print(dropdown_menu_html,'\n')
        with open(file['html_path'], 'w') as f:
            f.write(output)

    GenerateMainPage(LangLists, processed_files, dropdown_menu_html)

if __name__ == "__main__":
    main()