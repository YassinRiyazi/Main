import importlib.util
import os
import urllib.request
import socket

def load_remote_module_to_temp(url, module_name="contVideo", timeout=5):
    """
    Downloads a Python module from GitHub into a temp folder relative to this script, and imports it.

    <h3>Install Python Package</h3>
    <pre><code class="language-bash">pip install load-remote-module</code></pre>
    
    Args:
        url (str): Raw GitHub URL of the Python module.
        module_name (str): Name to use when importing the module.
        timeout (int): Timeout in seconds for the internet connectivity check.

    Returns:
        module: The dynamically imported module.
    """

    def has_internet(host="8.8.8.8", port=53, timeout=timeout):
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error:
            return False

    # Define local temp folder relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(base_dir, "_temp_modules")
    os.makedirs(temp_dir, exist_ok=True)

    local_path = os.path.join(temp_dir, f"{module_name}.py")

    # Check internet and download latest file if online
    if has_internet():
        try:
            print(f"Downloading {module_name}.py from GitHub...")
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            print(f"Download failed: {e}")
            raise
    else:
        print("No internet connection. Using cached module if available.")

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Module not found at: {local_path}")

    # Dynamic import
    spec = importlib.util.spec_from_file_location(module_name, local_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__ == "__main__":
    github_raw_url = (
        "https://raw.githubusercontent.com/YassinRiyazi/Main/main/src/PyThon/ContinuousVideoExperiments/contVideo.py"
    )
    cont_video = load_remote_module_to_temp(github_raw_url)
    print(f"Module {cont_video.__name__} loaded successfully from {github_raw_url}")