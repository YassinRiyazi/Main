import re


def parse_c_comments(comment):
    """
    Parse a Doxygen-style C comment block into structured documentation fields.

    Supported tags:
        @brief, @param, @return, @note, @warning

    Args:
        comment (str): Raw comment string from a C file, including Doxygen tags.

    Returns:
        dict: A dictionary with the following structure:
            {
                'brief': str,
                'description': str,
                'params': list of {'name': str, 'desc': str},
                'returns': {'type': str, 'desc': str} or None,
                'notes': str,
                'warnings': str
            }

    Example:
        >>> comment = '''
        ... /**
        ...  * @brief Adds two integers.
        ...  * @param a First number
        ...  * @param b Second number
        ...  * @return int Sum of a and b
        ...  */
        ... '''
        >>> parsed = parse_c_comments(comment)
        >>> print(parsed['brief'])
        Adds two integers.
    """
    sections = {
        'brief': [],
        'description': [],
        'params': [],
        'returns': None,
        'notes': [],
        'warnings': []
    }
    current_section = 'description'
    
    for line in comment.split('\n'):
        line = line.strip(' *')
        if not line:
            continue
            
        # Check for tags
        if line.startswith('@brief'):
            current_section = 'brief'
            line = line[6:].strip()
        elif line.startswith('@param'):
            current_section = 'params'
            line = line[6:].strip()
        elif line.startswith('@return'):
            current_section = 'returns'
            line = line[7:].strip()
        elif line.startswith('@note'):
            current_section = 'notes'
            line = line[5:].strip()
        elif line.startswith('@warning'):
            current_section = 'warnings'
            line = line[8:].strip()
            
        # Parse params
        if current_section == 'params':
            match = re.match(r'(\w+)\s+(.*)', line)
            if match:
                sections['params'].append({
                    'name': match.group(1),
                    'desc': match.group(2)
                })
        # Parse returns
        elif current_section == 'returns':
            match = re.match(r'([^\s]+)\s+(.*)', line)
            if sections['returns'] is None:
                if match:
                    sections['returns'] = {
                        'type': match.group(1),
                        'desc': match.group(2)
                    }
                else:
                    sections['returns'] = {
                        'type': line,
                        'desc': ''
                    }
        # Other sections
        else:
            sections[current_section].append(line)
    
    # Join multi-line sections
    for section in ['brief', 'description', 'notes', 'warnings']:
        sections[section] = ' '.join(sections[section])
    
    return sections

def extract_c_functions(file_path):
    """
    Extract C functions and their associated Doxygen comments from a C file.

    Only functions with /** ... */ style comments immediately preceding them are extracted.

    Args:
        file_path (str): Path to the C source file (.c or .h).

    Returns:
        list of dict: Each function is represented as:
            {
                'name': str,
                'return_type': str,
                'doc': dict  # Parsed output from parse_c_comments()
            }

    Example:
        >>> funcs = extract_c_functions("example.c")
        >>> print(funcs[0]['name'])
        my_function
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    pattern = re.compile(
        r'/\*\*(.*?)\*/\s*\n\s*(\w[\w\s]+\*?)\s+(\w+)\s*\([^)]*\)\s*\{',
        re.DOTALL
    )
    
    functions = []
    for match in pattern.finditer(content):
        comment = match.group(1).strip()
        return_type = match.group(2).strip()
        func_name = match.group(3).strip()
        
        doc_sections = parse_c_comments(comment)
        functions.append({
            'name': func_name,
            'return_type': return_type,
            'doc': doc_sections
        })
    
    return functions


def generate_c_function_html(func):
    """
    Generate HTML documentation for a single C function.

    Args:
        func (dict): A function dictionary returned from `extract_c_functions`, with keys:
            - 'name' (str): Function name.
            - 'return_type' (str): Return type of the function.
            - 'doc' (dict): Documentation sections from `parse_c_comments`.

    Returns:
        str: HTML string describing the function in a structured format.

    Example:
        >>> html = generate_c_function_html(funcs[0])
        >>> print(html)
    """
    doc = func['doc']
    html = [
        '<div class="function">',
        f'<div class="function-name">{func["return_type"]} {func["name"]}()</div>',
        f'<div class="brief">{doc["brief"]}</div>',
        f'<div class="description">{doc["description"]}</div>'
    ]
    if doc['params']:
        html.append('<div class="section-title">Parameters:</div><ul>')
        for param in doc['params']:
            html.append(f'<li><code>{param["name"]}</code>: {param["desc"]}</li>')
        html.append('</ul>')
    if doc['returns']:
        html.append('<div class="section-title">Returns:</div>')
        html.append(f'<div><code>{doc["returns"]["type"]}</code>: {doc["returns"]["desc"]}</div>')
    html.append('</div>')
    return '\n'.join(html)