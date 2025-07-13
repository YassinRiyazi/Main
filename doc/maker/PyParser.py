import re

def parse_python_docstring(docstring):
    """Parse a Python docstring into a structured dictionary."""
    sections = {
        'description': [],
        'args': [],
        'returns': None,
        'raises': [],
        'examples': []
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
        # Default to description
        else:
            sections['description'].append(line)
    
    # Join description lines
    sections['description'] = ' '.join(sections['description'])
    return sections


def extract_python_functions(file_path):
    """Extract functions and their docstrings from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    pattern = re.compile(
        r'def\s+(\w+)\s*\([^)]*\)\s*:\s*\n\s*"""(.*?)"""',
        re.DOTALL
    )
    
    functions = []
    for match in pattern.finditer(content):
        func_name = match.group(1)
        docstring = match.group(2).strip()
        doc_sections = parse_python_docstring(docstring)
        functions.append({
            'name': func_name,
            'doc': doc_sections
        })
    
    return functions