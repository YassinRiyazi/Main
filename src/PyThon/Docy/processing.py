import os
import glob

def css_styles(file,rootDir = 'docs'):
    """
        Walking over CSS files,
        calculating relative path to each HTML and place it inside HTML with proper padding.
    """
    _css_links = ''
    for ccs in glob.glob(os.path.join(rootDir,'styles','*.css')):
        relative_path = os.path.relpath(ccs, start=os.path.dirname(file['html_path']))
        _css_links += f'            <link rel="stylesheet" href="{relative_path}">\n'
    return _css_links