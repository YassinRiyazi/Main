import re


def parse_c_comments(comment):
    """Parse C-style Doxygen comments into a structured dictionary."""
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
    """Extract functions and their comments from a C file."""
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