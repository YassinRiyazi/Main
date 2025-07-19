import      natsort
from        .cv2_util      import *
from        .ffmpeg_util   import *
from        .Shumaly_util  import *
from        .visualization import *

def get_subdirectories(root_dir, max_depth=2):
    directories = []
    for root, dirs, _ in sorted(os.walk(root_dir)):
        if root == root_dir:
            continue  # Skip the root directory itself
        depth = root[len(root_dir):].count(os.sep)
        if depth < max_depth:
            directories.append(root)
        else:
            del dirs[:]  # Stop descending further
    return directories

def load_files(ad):
    valid_extensions = {"tiff", "tif", "png", "jpg", "jpeg", "bmp", "gif", "webp"}  # Common image formats
    FileNames = []
    for file in sorted(os.listdir(ad)):
        try:
            if file.split(".")[-1].lower() in valid_extensions:
                FileNames.append(file)
        except IndexError:
            pass
    return natsort.natsorted(FileNames)

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that the specified directory exists. 
    If it does not exist, create it along with any necessary parent directories.

    :param directory: Path of the directory to check and create if necessary.
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")

def writter(_adress):
    with open(_adress, 'w') as output:
        pass