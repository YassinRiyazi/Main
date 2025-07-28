import re
import ast

def parse_python_docstring(docstring):
    """
    Parse a Google-style Python docstring into a structured dictionary.

    Recognized sections include: Args, Returns, Raises, Examples, Notes, See Also, Caution, and Warning.

    Args:
        docstring (str): A raw docstring from a Python function.

    Returns:
        dict: Structured documentation with keys:
            {
                'description': str,
                'args': list of {'name': str, 'type': str, 'desc': str},
                'returns': {'type': str, 'desc': str} or None,
                'raises': list of {'type': str, 'desc': str},
                'examples': list of str,
                'notes': list of str,
                'see_also': list of str,
                'caution': list of str,
                'warning': list of str,
            }

    Example:
        >>> doc = '''
        ...     Adds two numbers.
        ...     Args:
        ...         a (int): First number.
        ...         b (int): Second number.
        ...     Returns:
        ...         int: Sum of the numbers.
        ...     '''
        >>> parse_python_docstring(doc)
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
    """Extract top-level Python functions and their docstrings from a source file.

    Uses the AST (Abstract Syntax Tree) module to safely parse function names and docstrings.

    Args:
        file_path (str): Path to the Python source file (.py).

    Returns:
        list of dict: Each function is represented as:
            {
                'name': str,
                'doc': dict  # Parsed result from parse_python_docstring()
            }

    Example:
        >>> extract_python_functions("my_script.py")
        [{'name': 'foo', 'doc': {...}}, ...]
    """

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
    """Generate HTML representation of a Python function’s documentation.

    Args:
        func (dict): Function object from `extract_python_functions()` with keys:
            - 'name' (str): Function name.
            - 'doc' (dict): Parsed docstring data from `parse_python_docstring()`.

    Returns:
        str: HTML markup representing the function’s name, parameters, return type,
             and additional sections like notes, examples, and warnings.

    Example:
        >>> html = generate_python_function_html({'name': 'add', 'doc': {...}})
        >>> print(html)
    """

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
