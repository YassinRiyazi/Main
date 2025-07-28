import os
import glob

def css_styles(file, rootDir='docs'):
    """
    Generate relative <link> tags for all CSS files under a styles directory.

    This is used to inject CSS references into each HTML file based on its location.

    Args:
        file (dict): A dictionary representing the current file, containing 'html_path'.
        rootDir (str, optional): Root directory containing the 'styles' folder. Defaults to 'docs'.

    Returns:
        str: A string containing HTML <link> elements with relative paths to CSS files.

    Example:
        >>> css = css_styles({'html_path': 'docs/module/file.html'})
        >>> print(css)
        <link rel="stylesheet" href="../styles/theme.css">
    """

    _css_links = ''
    for ccs in glob.glob(os.path.join(rootDir,'styles','*.css')):
        relative_path = os.path.relpath(ccs, start=os.path.dirname(file['html_path']))
        _css_links += f'            <link rel="stylesheet" href="{relative_path}">\n'
    return _css_links