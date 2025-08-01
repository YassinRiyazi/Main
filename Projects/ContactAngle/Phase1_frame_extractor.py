import  os
import  tqdm
import  json
import  natsort
import  subprocess
import  numpy               as      np
import  matplotlib.pyplot   as      plt

from    skimage.measure     import  ransac, LineModelND

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

def get_video_dimensions(video_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")
    video_info = json.loads(result.stdout)
    width = video_info["streams"][0]["width"]
    height = video_info["streams"][0]["height"]
    return width, height


def img_mkr(experiment, use_select_filter=True):
    video = f"./{experiment}/{experiment.split('/')[-1]}.mp4"
    width, height = get_video_dimensions(video)
    _idx = 1024 - height

    # Base filter
    """
    Explanation of Your Padding (pad=iw:ih+20:0:20:black)
        iw (input width)                    → Keeps the width unchanged.
        ih+20 (input height + 20 pixels)    → Adds 20 extra pixels to the height.
        0 (x_offset)                        → The original video stays at the same horizontal position (not shifted left or right).
        20 (y_offset)                       → Moves the original video down by 20 pixels, so the extra space appears at the bottom.
        black                               → Fills the new padding area with black color.
    """
    filter_chain = f"pad=iw:ih+{_idx}:0:{_idx}:white"

    # Add the select filter if enabled
    if use_select_filter:
        filter_chain += ",select='lt(n\\,15)'"

    subprocess.run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", video,
        "-vf", filter_chain,
        "-vsync", "vfr",
        "-q:v", "2",
        f"{experiment}/%06d.jpg"
    ])

def img_mkr_rotation(experiment, N, angle):
    subprocess.run([
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", f"./{experiment}/{experiment.split('/')[-1]}.mp4",
                "-vf", f"crop=1280:ih-{N}:0:0,rotate={angle}*(PI/180):ow=rotw({angle}*(PI/180)):oh=roth({angle}*(PI/180)):c=white,select='lt(n\\,15)'",
                "-vsync", "vfr",
                "-q:v", "2",
                f"{experiment}/%06d.jpg"
    ])

if __name__ == "__main__":
    for tilt in get_subdirectories(r"Bubble"):
        for experiment in tqdm.tqdm(get_subdirectories(tilt)):
            img_mkr(experiment,use_select_filter=0)