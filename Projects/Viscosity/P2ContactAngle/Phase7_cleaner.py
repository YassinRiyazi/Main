import shutil
import os
import tqdm
import utils
import subprocess



def remove_jpg_in_directory_only(directory_path):
    """
    Removes all .jpg files in the specified directory, excluding subdirectories.

    Args:
        directory_path (str): Path to the directory to clean.
    """
    for item in os.listdir(directory_path):
        full_path = os.path.join(directory_path, item)
        if os.path.isfile(full_path) and item.lower().endswith('.jpg'):
            os.remove(full_path)
            # print(f"Deleted: {full_path}")

if __name__ == "__main__":
    experiments = []
        # ffmpeg command components
    


    for tilt in utils.get_subdirectories(r"Processed_bubble_unpadded"):
        for experiment in tqdm.tqdm(utils.get_subdirectories(tilt)):
            if not os.path.isfile(os.path.join(experiment,"SR_result","result.xlsx")):
                print(experiment)
                remove_jpg_in_directory_only(experiment)
            else:
                pass
                # remove_jpg_in_directory_only(experiment)
                # Generate a temporary filter file for drawtext (one filter for all)
                
                # Use ffmpeg's glob to match the frames
        #         cmd = [
        #             'ffmpeg',
        #             '-framerate', '24',
        #             '-pattern_type', 'glob',
        #             '-i', f'/media/ysn/DONT/S4S-ROF/{experiment}/SR_edge/*.jpg',
        #             '-vf', f'scale=1920:1280',  # Make width and height even
        #             '-c:v', 'libx264',
        #             '-pix_fmt', 'yuv420p',
        #             '-y',
        #             f'/media/ysn/DONT/S4S-ROF/{experiment}/SR_edge/output.mp4'
        #         ]

        #         # Run the command
        #         subprocess.run(cmd, check=True)
        #         remove_jpg_in_directory_only(os.path.join(experiment,"SR_edge"))
        # #     break
        # # break

