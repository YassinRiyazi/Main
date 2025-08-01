import re
import ast

def parse_python_docstring(docstring: str) -> dict:
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
                'TODO': list of str,
                'FIXME': list of str,
                'HACK': list of str,
                'XXX': list of str
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
        'TODO': [],
        'FIXME': [],
        'HACK': [],
        'XXX': [],
        'Task': [],
        'sub-task': [],
        'sub-sub-task': [],
        'sub-sub-sub-task': [],
        'sub-sub-sub-sub-task': [],
        'references': []    
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
        elif line.lower().startswith('todo:'):
            current_section = 'TODO'
            continue
        elif line.lower().startswith('fixme:'):
            current_section = 'FIXME'
            continue    
        elif line.lower().startswith('hack:'):
            current_section = 'HACK'
            continue    
        elif line.lower().startswith('xxx:'):
            current_section = 'XXX'
            continue 
        elif line.lower().startswith('task:'):
            current_section = 'Task'
            continue
        elif line.lower().startswith('sub-task:'):
            current_section = 'sub-task'
            continue    
        elif line.lower().startswith('sub-sub-task:'):
            current_section = 'sub-sub-task'
            continue
        elif line.lower().startswith('sub-sub-sub-task:'):
            current_section = 'sub-sub-sub-task'
            continue    
        elif line.lower().startswith('sub-sub-sub-sub-task:'):
            current_section = 'sub-sub-sub-sub-task'
            continue
        elif line.lower().startswith('references:'):
            current_section = 'references'
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
                    'name': match.group(1).strip(),
                    'desc': match.group(2).strip()
                }
        # Parse raises section
        elif current_section == 'raises':
            match = re.match(r'(\w+):\s*(.*)', line)
            if match:
                sections['raises'].append({
                    'name': match.group(1),
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

        elif current_section == 'TODO':
            sections['TODO'].append(line)

        elif current_section == 'FIXME':
            sections['FIXME'].append(line)

        elif current_section == 'HACK':
            sections['HACK'].append(line)

        elif current_section == 'XXX':
            sections['XXX'].append(line)
        
        elif current_section == 'Task':
            sections['Task'].append(line)

        elif current_section == 'sub-task':
            sections['sub-task'].append(line)

        elif current_section == 'sub-sub-task':
            sections['sub-sub-task'].append(line)

        elif current_section == 'sub-sub-sub-task':
            sections['sub-sub-sub-task'].append(line)

        elif current_section == 'sub-sub-sub-sub-task':
            sections['sub-sub-sub-sub-task'].append(line)
        
        elif current_section == 'references':
            sections['references'].append(line)



        else:
            sections['description'].append(line)
    
    # Join description lines
    sections['description'] = ' '.join(sections['description'])
    return sections

def generate_python_function_html(obj: dict, indent: int = 1) -> str:
    """
    Generate indented HTML for a Python function or class’s documentation.
    Args:
        obj (dict): A dictionary representing a Python function or class with its docstring.
        indent (int): The indentation level for the HTML output.
    Returns:
        str: The generated HTML string.
    """
    def indent_line(line, level):
        return '      ' * level + line

    lines = []
    level = indent

    if obj['type'] == 'function':
        doc = obj['doc']
        args_string = ', '.join(
            f'<span style="color:#94D6BFFF;"><b>{arg["name"]}</b></span>:<span style="color:#42C39DFF;"><b>{arg["type"]}</b></span>'
            for arg in doc['args']
        )

        lines.append(indent_line('<div class="function">', level))
        lines.append(indent_line(f'<div class="function-name">{obj["name"]}({args_string})</div>', level + 1))
        lines.append(indent_line(f'<div class="description">{doc["description"]}</div>', level + 1))

        if doc['args']:
            lines.append(indent_line('<div class="section-title">Parameters:</div>', level + 1))
            lines.append(indent_line('<ul>', level + 1))
            for arg in doc['args']:
                lines.append(indent_line(f'<li><code>{arg["name"]}</code> ({arg["type"]}): {arg["desc"]}</li>', level + 2))
            lines.append(indent_line('</ul>', level + 1))

        if doc['returns']:
            lines.append(indent_line('<div class="section-title">Returns:</div>', level + 1))
            lines.append(indent_line(f'<div><code>{doc["returns"]["name"]}</code>: {doc["returns"]["desc"]}</div>', level + 2))

        if doc['raises']:
            lines.append(indent_line('<div class="section-title">Raises:</div>', level + 1))
            lines.append(indent_line('<ul>', level + 1))
            for exc in doc['raises']:
                lines.append(indent_line(f'<li><code style="color:red;">{exc["name"]}</code>: {exc["desc"]}</li>', level + 2))
            lines.append(indent_line('</ul>', level + 1))

        for key in doc:
            if key not in ['args', 'description', 'returns', 'raises'] and doc[key]:
                lines.append(indent_line(f'<div class="section-title" id="{obj["name"]}-{key}">{key.capitalize()}:</div>', level + 1))
                for item in doc[key]:
                    lines.append(indent_line(f'<div>{item}</div>', level + 2))

        lines.append(indent_line('</div>', level))  # end function

    #########################################################################################################
    elif obj['type'] == 'class':
        doc = obj['doc']
        lines.append(indent_line('<div class="class">', level))
        lines.append(indent_line(f'<div class="class-name">class <b>{obj["name"]}</b></div>', level + 1))
        lines.append(indent_line(f'<div class="description">{doc.get("description", "")}</div>', level + 1))

        for key in doc:
            if key != 'description' and doc[key]:
                lines.append(indent_line(f'<div class="section-title">{key.capitalize()}:</div>', level + 1))
                for item in doc[key]:
                    lines.append(indent_line(f'<div>{item}</div>', level + 2))

        if obj.get('methods'):
            lines.append(indent_line('<div class="section-title">Methods:</div>', level + 1))
            for method in obj['methods']:
                lines.append(indent_line('<div class="method">', level + 1))
                lines.append(generate_python_function_html({
                    'type': 'function',
                    'name': method['name'],
                    'doc': method['doc']
                }, indent=level + 2))
                lines.append(indent_line('</div>', level + 1))

        lines.append(indent_line('</div>', level))  # end class

    return '\n'.join(lines)



def extract_python_objects(file_path):
    """Extract Python functions and classes (with methods) and their docstrings from a source file.

    Args:
        file_path (str): Path to the Python source file (.py).

    Returns:
        list of dict: Each object is either a function or class represented as:
            {
                'type': 'function' or 'class',
                'name': str,
                'doc': dict,  # Parsed docstring
                'methods': list (only for class): [
                    {
                        'name': str,
                        'doc': dict  # Parsed method docstring
                    }
                ]
            }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=file_path)

    objects = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            if docstring:
                doc_sections = parse_python_docstring(docstring)
                objects.append({
                    'type': 'function',
                    'name': node.name,
                    'doc': doc_sections
                })
        elif isinstance(node, ast.ClassDef):
            class_doc = ast.get_docstring(node)
            class_info = {
                'type': 'class',
                'name': node.name,
                'doc': parse_python_docstring(class_doc) if class_doc else {},
                'methods': []
            }
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef):
                    method_doc = ast.get_docstring(body_item)
                    if method_doc:
                        class_info['methods'].append({
                            'name': body_item.name,
                            'doc': parse_python_docstring(method_doc)
                        })
            objects.append(class_info)

    return objects

if __name__ == "__main__":
    items = extract_python_objects("src/PyThon/Test/test.py")
    for obj in items:
        if obj['type'] == 'function':
            # print(generate_python_function_html(obj))
            pass
        elif obj['type'] == 'class':
            print(f"<h2>Class: {obj['name']}</h2>")
            print(f"<p>{obj['doc']['description']}</p>")
            for method in obj['methods']:
                print(generate_python_function_html(method))