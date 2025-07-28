import os
import re
from bs4 import BeautifulSoup

############# Type 3 refrence [File name] #############
def fileNameExtractor_Langless(processed_files, source_dirs):
    file_name_to_html_path = {}

    # Populate file name to HTML path mapping for each language
    for file in processed_files:
        file_name_to_html_path[file['name']] = file['html_path']

    return file_name_to_html_path

def fileNameExtractor(processed_files, source_dirs):
    """
    \Label: fileNameExtractor

    Builds a mapping from file names to their corresponding HTML paths,
    grouped by language.

    This is later used by \Ref: replace_file_names_in_html to hyperlink
    file name mentions in HTML content.
    
    Args:
        processed_files (list[dict]): List of processed file metadata, where each item
            includes 'lang', 'name', and 'html_path'.
        source_dirs (dict): Dictionary mapping language codes to their source directories.

    Returns:
        dict: A nested dictionary structured as {lang: {file_name: html_path}}.
    """
    file_name_to_html_path = {}

    # Initialize empty dict for each language
    for lang in source_dirs.keys():
        file_name_to_html_path[lang] = {}

    # Populate file name to HTML path mapping for each language
    for file in processed_files:
        file_name_to_html_path[file['lang']][file['name']] = file['html_path']

    return file_name_to_html_path

def replace_file_names_in_html(html_content, file_name_to_html_path, current_html_path):
    """
    \Label: replace_file_names_in_html

    Replaces plain file name mentions in HTML content with relative hyperlinks to
    their corresponding HTML pages, except in code blocks or existing anchor tags.

    Args:
        html_content (str): The raw HTML content as a string.
        file_name_to_html_path (dict): Dictionary mapping file names to their target HTML paths.
        current_html_path (str): Path to the HTML file being processed (used for relative linking).

    Returns:
        str: Modified HTML content with appropriate <a> tags added.
    """
    if not file_name_to_html_path:
        return html_content

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Build a regex pattern that matches any file name exactly (word boundary)
    pattern = r'\b(' + '|'.join(map(re.escape, file_name_to_html_path.keys())) + r')\b'

    # Iterate through all text nodes in the HTML
    for text_node in soup.find_all(text=True):
        # Skip non-string nodes
        if not isinstance(text_node, str):
            continue

        parent = text_node.parent

        # Skip text inside <code>, <pre>, <a>, or any ancestor <a> tags
        if parent.name in ['code', 'pre', 'a'] or any(ancestor.name == 'a' for ancestor in parent.parents):
            continue

        def replace_match(match):
            word = match.group(0)
            if word == 'notes':
                return word  # Do not hyperlink the word 'notes'
            target_path = file_name_to_html_path[word]
            # Compute relative path from current file to target file
            rel_path = os.path.relpath(target_path, start=os.path.dirname(current_html_path))
            rel_path = rel_path.replace('\\', '/')  # Ensure URL-safe path
            # Return the hyperlink-wrapped word
            return f'<span class="rel-link"><a href="{rel_path}">{word}</a></span>'

        # Replace file name matches in the text node
        new_text = re.sub(pattern, replace_match, text_node)

        # Replace the original text with new HTML
        text_node.replace_with(BeautifulSoup(new_text, 'html.parser'))

    return str(soup)

############# Type 2 refrence [Function name Manual] #############
 

def process_html_for_labels(html_content):
    """
    Process HTML content to replace \Label tags with HTML anchors and collect all labels.

    This function scans the HTML for `\Label: <label_name>` patterns, replaces each
    with an HTML anchor (for linking), and returns the modified HTML content along
    with a list of the extracted labels.

    Args:
        html_content (str): Raw HTML string containing \Label tags.

    Returns:
        tuple:
            - processed_html (str): HTML with anchor tags inserted.
            - labels (list[str]): List of extracted label names.
    """
    labels = []

    def replace_label(match):
        label = match.group(1)
        labels.append(label)
        # Insert label visually and as an anchor target
        return f'<p> <span class="keyword">{label.split(r"\\Label: ")[0]}</span></p><span id="{label}"></span>'

    # Regex to find all \Label: label_name patterns
    processed_html = re.sub(r'\\Label:\s*(\w+)', replace_label, html_content)
    return processed_html, labels

# Second pass: Replace \Ref with hyperlinks
def process_html_for_labels_replace(processed_files: list, label_to_file: dict):
    """
    Replace \Ref tags in HTML files with hyperlinks to corresponding label anchors.

    For each HTML file in `processed_files`, this function reads the file, finds all
    `\Ref: <label_name>` tags, and replaces them with HTML links pointing to the file
    and anchor associated with that label (if available in `label_to_file`).

    Args:
        processed_files (list[dict]): List of processed file metadata dictionaries,
            each containing at least a 'html_path' key.
        label_to_file (dict): A mapping from label names to the HTML file path
            containing the corresponding anchor.
    """
    for file in processed_files:
        html_path = file['html_path']

        # Read the HTML file content
        with open(html_path, 'r') as f:
            content = f.read()

        def replace_ref(match):
            label = match.group(1)
            if label in label_to_file:
                target_path = label_to_file[label]
                # Compute relative path to the label’s file
                rel_path = os.path.relpath(target_path, start=os.path.dirname(html_path))
                rel_path = rel_path.replace('\\', '/')  # Normalize Windows paths for web
                # Create clickable link to label anchor
                return f'<p> <span class="keyword-ref"><a href="{rel_path}#{label}">{label}</a></span> </p>'
            else:
                print(f"Warning: undefined reference '{label}' in {html_path}")
                # Leave the original text unchanged if label is not found
                return match.group(0)

        # Replace all \Ref: label_name tags in the HTML
        content = re.sub(r'\\Ref:\s*(\w+)', replace_ref, content)

        # Write the modified content back to the file
        with open(html_path, 'w') as f:
            f.write(content)
