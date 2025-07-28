import re
import ast

def parse_python_docstring(docstring):
    """
        Parse a Python docstring into a structured dictionary.
    """
    sections = {
        'description': [],
        'args': [],
        'returns': None,
        'raises': [],
        'examples': [],
        'notes': [],
        'see_also': [],
        'caution': [],
        'warning': [],
    }
    current_section = 'description'
    
    for line in docstring.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if line.lower().startswith('args:') or line.lower().startswith('parameters:'):
            current_section = 'args'
            continue
        elif line.lower().startswith('returns:'):
            current_section = 'returns'
            continue
        elif line.lower().startswith('raises:'):
            current_section = 'raises'
            continue
        elif line.lower().startswith('example:') or line.lower().startswith('examples:'):
            current_section = 'examples'
            continue

        elif line.lower().startswith('notes:'):
            current_section = 'notes'
            continue
        elif line.lower().startswith('see_also:') or line.lower().startswith('see also:'):
            current_section = 'see_also'
            continue    
        elif line.lower().startswith('caution:'):
            current_section = 'caution'
            continue 
        elif line.lower().startswith('warning:'):
            current_section = 'warning'
            continue    
            
        # Parse args section
        if current_section == 'args':
            match = re.match(r'(\w+)\s*\(([^)]+)\):\s*(.*)', line)
            if match:
                sections['args'].append({
                    'name': match.group(1),
                    'type': match.group(2),
                    'desc': match.group(3)
                })
        # Parse returns section
        elif current_section == 'returns':
            match = re.match(r'([^:]+):\s*(.*)', line)
            if match:
                sections['returns'] = {
                    'type': match.group(1).strip(),
                    'desc': match.group(2).strip()
                }
        # Parse raises section
        elif current_section == 'raises':
            match = re.match(r'(\w+):\s*(.*)', line)
            if match:
                sections['raises'].append({
                    'type': match.group(1),
                    'desc': match.group(2)
                })
        # Parse examples section
        elif current_section == 'examples':
            sections['examples'].append(line)

        elif current_section == 'notes':
            sections['notes'].append(line)

        elif current_section == 'see_also':
            sections['see_also'].append(line)

        elif current_section == 'caution':
            sections['caution'].append(line)

        elif current_section == 'warning':
            sections['warning'].append(line)

        else:
            sections['description'].append(line)
    
    # Join description lines
    sections['description'] = ' '.join(sections['description'])
    return sections


def extract_python_functions(file_path):
    """Extract functions and their docstrings from a Python file using AST."""
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=file_path)
    
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            if docstring:
                doc_sections = parse_python_docstring(docstring)
                functions.append({
                    'name': node.name,
                    'doc': doc_sections
                })
    return functions

def generate_python_function_html(func):
    """Generate HTML for a Python function."""
    doc = func['doc']

    vv = ""
    for arg in doc['args']:
            vv += f'<span style="color:#94D6BFFF;"><b>{arg["name"]}</b></span>:<span style="color:#42C39DFF;"><b>{arg["type"]}</b></span>, '

    html = [
        '<div class="function">',
        f'<div class="function-name">{func["name"]}({vv[:-1]})</div>',
        f'<div class="description">{doc["description"]}</div>'
    ]

    if doc['args']:
        html.append('<div class="section-title">Parameters:</div><ul>')
        for arg in doc['args']:
            html.append(f'<li><code>{arg["name"]}</code> ({arg["type"]}): {arg["desc"]}</li>')
        html.append('</ul>')

    if doc['returns']:
        html.append('<div class="section-title">Returns:</div>')
        html.append(f'<div><code>{doc["returns"]["type"]}</code>: {doc["returns"]["desc"]}</div>')

    for kk in doc:
        if doc[kk] != [] and kk != 'args' and kk != 'description' and kk != 'returns':
            html.append(f'<div class=section-title>{kk}:</div>')
            for i in doc[kk]:
                html.append(f'<div>{i}</div>')

    html.append('</div>')
    return '\n'.join(html)


if __name__ == "__main__":
    print(extract_python_functions("src/PyThon/Test/test.py"))
