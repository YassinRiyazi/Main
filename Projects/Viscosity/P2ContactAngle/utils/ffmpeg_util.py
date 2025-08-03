import subprocess
import os
import numpy as np
import json

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


def img_mkr(video, use_select_filter=True,padding=False,
            rotate=0,):

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
    
    if rotate!=0:
        rotate = rotate*(np.pi/180)
        filter_chain += f"rotate={rotate}:ow=rotw({rotate}):oh=roth({rotate}):c=white"

    # Prepare the command list
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        # "-hwaccel", "cuda",
        "-i", video
    ]
    
    # Add the -vf flag only if padding is enabled or any filters are used
    if padding or use_select_filter or rotate != 0:
        cmd += ["-vf", filter_chain]

    output_dir = os.path.join(video[:-4])
    output_dir = output_dir.replace("drop", "frames")  # Replace spaces with underscores
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # print(f"Created directory: {output_dir}")
    
    # Add other arguments
    cmd += [
        "-vsync", "vfr",
        "-q:v", "2",
        f"{output_dir}/%06d.jpg"
    ]


    
    # Run the command
    subprocess.run(cmd)



def create_video_from_images(image_folder, output_video_path, fps=30):
    # # Get all .jpg files from the folder
    # images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # # Sort images to maintain the sequence
    # images.sort()

    # # Check if there are images in the folder
    # if not images:
    #     print("No .jpg images found in the folder.")
    #     return

    # Change directory to the folder containing the images
    os.chdir(image_folder)

    # Construct the ffmpeg command to create a video from images
    # The images must be named in a sequential order, like image1.jpg, image2.jpg, etc.
    # FFmpeg uses a pattern such as 'image%d.jpg' to match the files
    command = [
        'ffmpeg',
        '-framerate',   str(fps),               # Set frames per second
        '-pattern_type', 'glob', '-i', '*.jpg', # Use 6-digit pattern to match the images (000612.jpg, 000613.jpg, ...)
        '-c:v',         'libx264',              # Video codec
        # '-pix_fmt',     'yuv420p',              # Pixel format for compatibility
        '-crf',         '18',                   # Set Constant Rate Factor for high quality (lower values = higher quality, 18-23 is typical range)
        '-preset',      'slow',                 # Use 'slow' preset for better compression and quality (other options: veryfast, fast, medium, slow, veryslow)
        '-tune',        'film',                 # Tune the encoding for film (preserves quality)
        '-y',
        output_video_path
    ]

    # Execute the command using subprocess
    try:
        subprocess.run(command, check=True)
        print(f"Video saved at {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video creation: {e}")


"""
ffmpeg -i "$output_video" -q:v 2 "$frame_dir/frame_%06d.jpg"
ffmpeg -i S1_30per_T3_C001H001S0001.mp4 -vf "pad=1280:1024:0:872:white" -c:v libx265 -crf 10 -preset veryslow -c:a copy output.mp4

ffmpeg -i S1_30per_T1_C001H001S0001.mp4 -vf "pad=1280:1024:0:872:white" -q:v 2 %06d.jpg
"""

