import os
import subprocess
from multiprocessing import Pool, cpu_count
import glob

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
        '-loglevel', 'error',
        '-framerate',   str(fps),               # Set frames per second
        '-pattern_type', 'glob', '-i', '*.jpg', # Use 6-digit pattern to match the images (000612.jpg, 000613.jpg, ...)
        '-c:v',         'libx264',              # Video codec
        # '-pix_fmt',     'yuv420p',              # Pixel format for compatibility
        '-crf',         '18',                   # Set Constant Rate Factor for high quality (lower values = higher quality, 18-23 is typical range)
        '-preset',      'slow',                 # Use 'slow' preset for better compression and quality (other options: veryfast, fast, medium, slow, veryslow)
        # '-tune',        'film',                 # Tune the encoding for film (preserves quality)
        '-y',
        output_video_path
    ]

    # Execute the command using subprocess
    try:
        subprocess.run(command, check=True)
        print(f"Video saved at {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video creation: {e}")

def get_mp4_files(root_dir, max_depth=2):
    mp4_files = []
    
    def scan_directory(path, current_depth=0):
        if current_depth >= max_depth:
            return
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith('.mp4'):
                            mp4_files.append(entry.path)  # Store full path
                        elif entry.is_dir(follow_symlinks=False) and current_depth + 1 < max_depth:
                            scan_directory(entry.path, current_depth + 1)
                    except (PermissionError, OSError):
                        continue  # Skip entries with access issues
        except (PermissionError, OSError) as e:
            print(f"Error accessing {path}: {e}")

    scan_directory(root_dir)
    return sorted(mp4_files)

def process_experiment(_adress):
    image_folder    = os.path.join(_adress, "SR_edge")
    outputpath      = os.path.join(image_folder, "result.mp4")
    result_csv      = os.path.join(_adress, 'SR_result', 'result.csv')

    # Skip if result.mp4 exists and is non-zero in size
    if os.path.isfile(outputpath) and os.path.getsize(outputpath) > 0:
        return

    # Check for images and CSV
    imgs = glob.glob(os.path.join(image_folder, "*.jpg"))
    if len(imgs) == 0 or not os.path.isfile(result_csv):
        return

    create_video_from_images(image_folder, outputpath)

    # Remove .jpg files after video is created
    for img_path in imgs:
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"⚠️ Error deleting {img_path}: {e}")

def process_experiment_frames(_adress):
    outputpath = os.path.join(_adress, "result.mp4")

    # Skip if result.mp4 exists and is non-zero in size
    if os.path.isfile(outputpath) and os.path.getsize(outputpath) > 0:
        return
    
    # Check for images and CSV
    imgs = glob.glob(os.path.join(_adress, "*.jpg"))
    if len(imgs) == 0:
        return

    create_video_from_images(_adress, outputpath)

    # Remove .jpg files after video is created
    for img_path in imgs:
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"⚠️ Error deleting {img_path}: {e}")


if __name__ == "__main__":
    adress = "/media/ubun25/DONT/MPI/S4S-ROF/drop/"
    experiments = sorted(get_mp4_files(adress, max_depth=5))
    experiments = [experiment.replace("drop","frame_Extracted") for experiment in experiments]
    experiments = [experiment.replace(".mp4","") for experiment in experiments]
    experiments = [experiment.replace("frames","frame_Extracted") for experiment in experiments]

    # ## Frames
    # experiments = sorted(get_mp4_files(adress, max_depth=5))
    # experiments = [experiment.replace("drop","frames") for experiment in experiments]
    # experiments = [experiment.replace(".mp4","") for experiment in experiments]
    # 
    print(experiments[0])
    with Pool(processes=cpu_count()//2) as pool:
        pool.map(process_experiment, experiments)

# /media/ubun25/DONT/MPI/S4S-ROF/frames/280/S2-SNr2.14_D/frames20250620_222000_DropNumber_01
# /media/ubun25/DONT/MPI/S4S-ROF/frames/280/S2-SNr2.14_D/frames20250620_222000_DropNumber_01