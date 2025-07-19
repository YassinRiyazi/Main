import utils
import tqdm
import os

if __name__ == "__main__":
    mainFolder = r"frames"
    main_folder_name = "Processed"
    for tilt in utils.get_subdirectories(mainFolder):
        for fluid in utils.get_subdirectories(tilt):
            for video in utils.get_subdirectories(fluid):
                video = video.replace(mainFolder, main_folder_name)  # Replace spaces with underscores
                if not os.path.exists(video):
                    os.makedirs(video, exist_ok=True)