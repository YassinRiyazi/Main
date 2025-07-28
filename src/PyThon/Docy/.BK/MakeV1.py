import os
import shutil
from datetime import datetime
import markdown
import Docy  # Assuming this module provides extract_python_functions and extract_c_functions
import re
from bs4 import BeautifulSoup

# Define source directories
source_dirs = {
    'Python': 'src/PyThon',
    'C': 'src/C',
    'C++': 'src/CPP',
    'CUDA': 'src/CUDA',
    'Devlog': 'doc/Devlog',
    'WebLog': 'doc/WebLog'
}
notes_dir       = 'doc/notes'
devlog_dir      = 'doc/Devlog'
weblog_dir      = 'doc/WebLog'
output_base_dir = 'docs'

def get_language_from_path(html_path, output_base_dir):
    """Determine the language from the HTML path."""
    rel_path = os.path.relpath(html_path, output_base_dir)
    parts = rel_path.split(os.sep)
    if len(parts) >= 2 and parts[0] in source_dirs.keys():
        return parts[0]
    return None

def process_html_for_labels(html_content):
    """Process HTML content to replace \label tags with anchors and collect labels."""
    labels = []
    def replace_label(match):
        label = match.group(1)
        labels.append(label)
        return f'<p> <span class="keyword">{label.split(r"\\Label: ")[0]}</span></p><span id="{label}"></span>'
    processed_html = re.sub(r'\\Label:\s*(\w+)', replace_label, html_content)
    return processed_html, labels

def generate_content_html(file_path):
    """Generate HTML content for source files and process labels."""
    if file_path.endswith('.py') and not file_path.endswith("__init__.py"):
        functions = Docy.extract_python_functions(file_path)
        content = [Docy.generate_python_function_html(func) for func in functions]
        lang = 'Python'
    elif file_path.endswith('.c'):
        functions = Docy.extract_c_functions(file_path)
        content = [Docy.generate_c_function_html(func) for func in functions]
        lang = 'C'
    else:
        return None, None, []
    content_html = '\n'.join(content) if content else '<p>No functions found.</p>'
    content_html, labels = process_html_for_labels(content_html)
    return content_html, lang, labels

def markdown2HTML(file_path):
    """Convert markdown to HTML."""
    with open(file_path, 'r') as f:
        markdown_content = f.read()
    html_content = markdown.markdown(markdown_content)
    return html_content

def generate_notes_html(file_path, lang):
    """Convert markdown notes to HTML and process labels."""
    if not os.path.exists(file_path):
        return f'<p>No {lang} notes available.</p>', lang, []
    html_content = markdown2HTML(file_path)
    html_content, labels = process_html_for_labels(html_content)
    return html_content, lang, labels

def build_tree(files, source_dir):
    """Build a nested dictionary for navigation structure."""
    tree = {}
    for file in files:
        if file['type'] == 'notes':
            tree['Notes'] = file['html_path']
        elif file['type'] == 'Devlog':
            tree[file['name']] = file['html_path']
        elif file['type'] == 'WebLog':
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
    html = ['<ul style="margin-left: {}px;">'.format(indent_level * 20)]
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
    
    lang_colors = {
        'Python': '#3572A5',
        'C': '#555555',
        'C++': '#F34B7D',
        'CUDA': '#3A4E3A',
        'Devlog': '#8B008B',
        'WebLog': '#076E75'
    }
    
    nav_html = ['<nav class="sidebar">', '<h2>Navigation</h2>', '<ul>']
    for lang in source_dirs.keys():
        if lang in languages and languages[lang]:
            open_attr = ' open' if current_file and current_file['lang'] == lang else ''
            nav_html.append(f'<li class="language language-{lang.lower()}" style="border-left: 4px solid {lang_colors[lang]};"><details{open_attr}><summary>{lang}</summary>')
            tree = build_tree(languages[lang], source_dirs.get(lang, devlog_dir if lang == 'Devlog' else weblog_dir if lang == 'WebLog' else ''))
            tree_html = generate_tree_html(tree, current_file_path, 
                                        ['Notes'] if current_file and current_file['type'] == 'notes' else 
                                        [current_file['name']] if current_file and current_file['type'] in ['Devlog', 'WebLog'] else 
                                        os.path.relpath(current_file['source_path'], source_dirs[lang]).split(os.sep) if current_file and current_file['type'] == 'source' else [])
            nav_html.append(tree_html)
            nav_html.append('</details></li>')
    nav_html.extend(['</ul>', '</nav>'])
    return '\n'.join(nav_html)

def replace_file_names_in_html(html_content, file_name_to_html_path, current_html_path):
    """Replace file names in HTML content with hyperlinks to their corresponding pages."""
    if not file_name_to_html_path:
        return html_content
    soup = BeautifulSoup(html_content, 'html.parser')
    pattern = r'\b(' + '|'.join(map(re.escape, file_name_to_html_path.keys())) + r')\b'
    for text_node in soup.find_all(text=True):
        
        if not isinstance(text_node, str):
            continue
        parent = text_node.parent
        if parent.name in ['code', 'pre', 'a'] or any(ancestor.name == 'a' for ancestor in parent.parents):
            continue
        def replace_match(match):
            word = match.group(0)
            if (word == 'notes'):
                return f'{word}'
            target_path = file_name_to_html_path[word]
            rel_path = os.path.relpath(target_path, start=os.path.dirname(current_html_path))
            rel_path = rel_path.replace('\\', '/')
            return f'<span class="rel-link"><a href="{rel_path}">{word}</a></span>'
        new_text = re.sub(pattern, replace_match, text_node)
        text_node.replace_with(BeautifulSoup(new_text, 'html.parser'))
    return str(soup)

def GenerateMainPage(LangLists, processed_files):
    """Generate index.html with all sections."""
    index_content = ['<h1>Project Documentation</h1>', '<div class="overview">']
    for lang in LangLists + ['Devlog', 'WebLog']:
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

def Markdowns2HTML(processed_files, _adress, _type):
    """Convert markdown files to HTML and process labels."""
    if os.path.exists(_adress):
        for date_folder in os.listdir(_adress):
            date_path = os.path.join(_adress, date_folder)
            if os.path.isdir(date_path):
                readme_path = os.path.join(date_path, 'README.md')
                if os.path.exists(readme_path):
                    content_html = markdown2HTML(readme_path)
                    content_html, labels = process_html_for_labels(content_html)
                    html_path = os.path.join(output_base_dir, _type, date_folder + '.html')
                    os.makedirs(os.path.dirname(html_path), exist_ok=True)
                    processed_files.append({
                        'content_html': content_html,
                        'lang': _type,
                        'source_path': readme_path,
                        'html_path': html_path,
                        'name': date_folder,
                        'type': _type,
                        'labels': labels
                    })

def main():
    if os.path.isdir(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    LangLists = ['Python', 'C', 'C++', 'CUDA']
    processed_files = []
    label_to_file = {}

    # Process source files
    for lang, dir_path in source_dirs.items():
        if os.path.exists(dir_path):
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(('.py', '.c', '.cu', '.h', '.cpp', '.hpp')):
                        file_path = os.path.join(root, file)
                        content_html, file_lang, labels = generate_content_html(file_path)
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
                                'type': 'source',
                                'labels': labels
                            })
                            for label in labels:
                                if label in label_to_file:
                                    print(f"Warning: duplicate label '{label}' in {html_path} and {label_to_file[label]}")
                                label_to_file[label] = html_path

    # Process notes files
    for lang in LangLists:
        notes_file = os.path.join(notes_dir, f'notes_{lang.lower()}.md')
        content_html, file_lang, labels = generate_notes_html(notes_file, lang)
        html_path = os.path.join(output_base_dir, lang, 'notes.html')
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        processed_files.append({
            'content_html': content_html,
            'lang': file_lang,
            'source_path': notes_file,
            'html_path': html_path,
            'name': f'{lang} Notes',
            'type': 'notes',
            'labels': labels
        })
        for label in labels:
            if label in label_to_file:
                print(f"Warning: duplicate label '{label}' in {html_path} and {label_to_file[label]}")
            label_to_file[label] = html_path

    Markdowns2HTML(processed_files, devlog_dir, 'Devlog')
    Markdowns2HTML(processed_files, weblog_dir, 'WebLog')

    # Build file name to HTML path mapping
    file_name_to_html_path = {}
    for lang in LangLists + ['Devlog', 'WebLog']:
        file_name_to_html_path[lang] = {}
        for file in processed_files:
            if file['lang'] == lang:
                if file['type'] == 'source':
                    base_name = os.path.splitext(os.path.basename(file['source_path']))[0]
                    if base_name not in file_name_to_html_path[lang]:
                        file_name_to_html_path[lang][base_name] = file['html_path']
                    else:
                        print(f"Warning: duplicate file name '{base_name}' in language {lang}")
                elif file['type'] == 'notes':
                    file_name_to_html_path[lang]['notes'] = file['html_path']
                elif file['type'] in ['Devlog', 'WebLog']:
                    file_name_to_html_path[lang][file['name']] = file['html_path']

    _tempWebAdress = "https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/"
    # Generate HTML for all files with file name hyperlinks
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
        content_html = replace_file_names_in_html(file['content_html'], file_name_to_html_path.get(file['lang'], {}), file['html_path'])
        output = template.replace('<!-- TITLE -->', title)
        output = output.replace('<!-- NAVIGATION -->', nav_menu)
        output = output.replace('<!-- CONTENT -->', content_html)
        output = output.replace('<!-- GENERATION_DATE -->', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        output = output.replace('<!-- Adress -->', f'<a href="{_tempWebAdress}{file["source_path"]}">{_tempWebAdress}{file["source_path"]}</a>')
        with open(file['html_path'], 'w') as f:
            f.write(output)

    # Second pass: Replace \Ref with hyperlinks
    for file in processed_files:
        html_path = file['html_path']
        with open(html_path, 'r') as f:
            content = f.read()
        def replace_ref(match):
            label = match.group(1)
            if label in label_to_file:
                target_path = label_to_file[label]
                rel_path = os.path.relpath(target_path, start=os.path.dirname(html_path))
                rel_path = rel_path.replace('\\', '/')
                return f'<p> <span class="keyword-ref"><a href="{rel_path}#{label}">{label}</a></span> </p>'
            else:
                print(f"Warning: undefined reference '{label}' in {html_path}")
                return match.group(0)
        content = re.sub(r'\\Ref:\s*(\w+)', replace_ref, content)
        with open(html_path, 'w') as f:
            f.write(content)

    GenerateMainPage(LangLists, processed_files)

if __name__ == "__main__":
    main()