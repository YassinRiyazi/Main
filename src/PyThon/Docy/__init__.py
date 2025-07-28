from .PyParser import *
from .CParser import *
from .processing import *
from .hyperlinks import *
import markdown


def markdown2HTML(file_path):
    """Convert a Markdown file to HTML.

    Args:
        file_path (str): Path to the Markdown (.md) file.

    Returns:
        str: HTML content converted from the Markdown input.

    Example:
        >>> html = markdown2HTML("README.md")
        >>> print(html)
    """
    with open(file_path, 'r') as f:
        markdown_content = f.read()
    html_content = markdown.markdown(markdown_content)
    return html_content