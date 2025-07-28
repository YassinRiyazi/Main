from .PyParser import *
from .CParser import *
from .processing import *
from .hyperlinks import *
import markdown


def markdown2HTML(file_path):
    """Convert markdown to HTML."""
    with open(file_path, 'r') as f:
        markdown_content = f.read()
    html_content = markdown.markdown(markdown_content)
    return html_content