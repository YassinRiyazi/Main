import os
import shutil
from datetime import datetime
import markdown
import maker  # Assuming this module provides extract_python_functions and extract_c_functions

# Define source directories
source_dirs = {
    'Python': 'src/PyThon',
    'C': 'src/C',
    'C++': 'src/CPP',
    'CUDA': 'src/CUDA',
    'Devlog': 'doc/Devlog',
    'WebLog' : 'doc/WebLog'
}
notes_dir = 'doc/notes'
devlog_dir = 'doc/Devlog'
weblog_dir = 'doc/WebLog'
output_base_dir = 'docs'

def get_language_from_path(html_path, output_base_dir):
    """Determine the language from the HTML path."""
    rel_path = os.path.relpath(html_path, output_base_dir)
    parts = rel_path.split(os.sep)
    if len(parts) >= 2 and parts[0] in source_dirs.keys():#['Python', 'C', 'C++', 'CUDA', 'Devlog']
        return parts[0]
    return None

def generate_content_html(file_path):
    """Generate HTML content for source files."""
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

def markdown2HTML(file_path):
    """Convert devlog markdown to HTML."""
    with open(file_path, 'r') as f:
        markdown_content = f.read()
    html_content = markdown.markdown(markdown_content)
    return html_content

def generate_notes_html(file_path, lang):
    """Convert markdown notes to HTML."""
    if not os.path.exists(file_path):
        return f'<p>No {lang} notes available.</p>', lang
    html_content = markdown2HTML(file_path)
    return html_content, lang

def build_tree(files, source_dir):
    """Build a nested dictionary for navigation structure."""
    tree = {}
    for file in files:
        if file['type'] == 'notes':
            tree['Notes'] = file['html_path']
        elif file['type'] == 'devlog':
            tree[file['name']] = file['html_path']
        elif file['type'] == 'wevlog':
            tree[file['name']] = file['html_path']

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

def generate_tree_html(tree, current_file_path, current_path_parts=None, indent_level=1):
    """Generate HTML for navigation tree with collapsible sections and indentation."""
    if current_path_parts is None:
        current_path_parts = []
    html = ['<ul style="margin-left: {}px;">'.format(indent_level * 20)]  # Indent by 20px per level
    for name in sorted(tree.keys(), reverse=(tree.get('Notes') is None)):
        value = tree[name]
        if isinstance(value, dict):
            is_open = current_path_parts and name == current_path_parts[0]
            open_attr = ' open' if is_open else ''
            sub_html = generate_tree_html(value, current_file_path, current_path_parts[1:] if is_open else [], indent_level + 1)
            html.append(f'<li><details{open_attr}><summary>{name}</summary>{sub_html}</details></li>')
        else:
            rel_path = os.path.relpath(value, start=os.path.dirname(current_file_path))
            rel_path = rel_path.replace('\\', '/')
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
    
    # Define colors for each language/section
    lang_colors = {
        'Python': '#3572A5',  # Python blue
        'C': '#555555',       # C gray
        'C++': '#F34B7D',    # C++ pink
        'CUDA': '#3A4E3A',   # CUDA green
        'Devlog': '#8B008B',  # Devlog purple
        'WebLog': "#076E75"  # Devlog purple
    }
    
    nav_html = ['<nav class="sidebar">', '<h2>Navigation</h2>', '<ul>']
    for lang in source_dirs.keys():#['Python', 'C', 'C++', 'CUDA', 'Devlog']
        if lang in languages and languages[lang]:
            open_attr = ' open' if current_file and current_file['lang'] == lang else ''
            # Add language-specific class for styling
            nav_html.append(f'<li class="language language-{lang.lower()}" style="border-left: 4px solid {lang_colors[lang]};"><details{open_attr}><summary>{lang}</summary>')
            tree = build_tree(languages[lang], source_dirs.get(lang, devlog_dir if (lang == 'Devlog' or lang == 'Weblog') else ''))
            tree_html = generate_tree_html(tree, current_file_path, 
                                        ['Notes'] if current_file and current_file['type'] == 'notes' else 
                                        [current_file['name']] if current_file and current_file['type'] == 'devlog' else 
                                        os.path.relpath(current_file['source_path'], source_dirs[lang]).split(os.sep) if current_file and current_file['type'] == 'source' else [])
            nav_html.append(tree_html)
            nav_html.append('</details></li>')
    nav_html.extend(['</ul>', '</nav>'])
    return '\n'.join(nav_html)

def GenerateMainPage(LangLists, processed_files):
    """Generate index.html with all sections."""
    index_content = ['<h1>Project Documentation</h1>', '<div class="overview">']
    for lang in LangLists + ['Devlog']:
        lang_files = [f for f in processed_files if f['lang'] == lang]
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

    with open(os.path.join(output_base_dir, 'index.html'), 'w') as f:
        f.write(output)

def Markdowns2HTML(processed_files,_adress,_type):
    if os.path.exists(_adress):
        for date_folder in os.listdir(_adress):
            date_path = os.path.join(_adress, date_folder)
            if os.path.isdir(date_path):
                readme_path = os.path.join(date_path, 'README.md')
                if os.path.exists(readme_path):
                    content_html = markdown2HTML(readme_path)
                    html_path = os.path.join(output_base_dir, _type, date_folder + '.html')
                    os.makedirs(os.path.dirname(html_path), exist_ok=True)
                    processed_files.append({
                        'content_html': content_html,
                        'lang': _type,
                        'source_path': readme_path,
                        'html_path': html_path,
                        'name': date_folder,
                        'type': _type
                    })

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

    # Process notes files (Language Tips)
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

    Markdowns2HTML(processed_files,devlog_dir,'Devlog')
    Markdowns2HTML(processed_files,weblog_dir,'WebLog')

    _tempWebAdress = "https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/"
    # Generate HTML for all files
    for file in processed_files:
        with open('doc/template.html', 'r') as f:
            template = f.read()
        nav_menu = create_nav_menu(processed_files, file['html_path'])
        if file['type'] == 'source':
            title = f"{file['name']} Documentation"

        elif file['type'] == 'notes':
            title = f"{file['lang']} Notes"
            
        elif file['type'] == 'Devlog':
            title = f""

        elif file['type'] == 'WebLog':
            title = f""

        output = template.replace('<!-- TITLE -->', title)
        output = output.replace('<!-- NAVIGATION -->', nav_menu)
        output = output.replace('<!-- CONTENT -->', file['content_html'])
        output = output.replace('<!-- GENERATION_DATE -->', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        output = output.replace('<!-- Adress -->', f'<a href="{_tempWebAdress}{file["source_path"]}">{_tempWebAdress}{file["source_path"]}</a>')
        with open(file['html_path'], 'w') as f:
            f.write(output)

    GenerateMainPage(LangLists, processed_files)

if __name__ == "__main__":
    main()