To Dynamically laod the src modules. 

TODO:
    For laoding the library build them and install them with Pip

```
    $   pip install build twine
```
# load-remote-module

Download and import remote Python modules (e.g., from GitHub) into a temporary folder.

## Usage

```python
from load_remote_module import load_remote_module_to_temp

mod = load_remote_module_to_temp("https://raw.githubusercontent.com/user/repo/branch/path/module.py")
