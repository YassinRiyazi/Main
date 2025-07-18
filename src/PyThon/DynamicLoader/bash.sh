python -m build

twine upload dist/*

pip install load-remote-module

>>>from load_remote_module import load_remote_module_to_temp
