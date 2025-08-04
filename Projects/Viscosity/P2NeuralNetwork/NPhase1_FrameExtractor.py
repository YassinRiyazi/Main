"""
    # -*- coding: utf-8 -*-
    Author: Yassin Riyazi
    Date: 03-07-2025

    
"""
import os
import tqdm
import glob

# Add the absolute path to the ./src folder
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'src/PyThon/FFMpeg')))
from Vidoe2Jpg import ffmpeg_frame_extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'src/PyThon/Utils')))
from FileFolder import get_subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'src/PyThon/OpenCV2')))
from videoHealth import check_videos

if __name__ == "__main__":
    """
    Main function to extract frames from videos.
    TODO:
        [V] Checking health of all videos
        [ ] Extracting frames from first 5 videos of each repetition at 15 fps, Later All 30fps
    """
    fps = 15

    VideosAddress = []
    for tilt in glob.glob("/media/d2u25/Dont/Teflon_VideoProcess/*"):
        for experiment in glob.glob(os.path.join(tilt,'*')):
            for _idx, rep in enumerate(glob.glob(os.path.join(experiment,'*','result.mp4'))):
                if _idx < 5:
                    VideosAddress.append(rep)

    VideosAddress, _, _ = check_videos(VideosAddress, remove_corrupted=False, verbose=False)
    print(f"Total healthy videos: {len(VideosAddress)}")

    for video_path in tqdm.tqdm(VideosAddress, desc="Extracting frames"):
        _temp = os.path.split(video_path)
        frame_path = os.path.join(_temp[0], 'frame_%06d.png')
        frame_path = frame_path.replace('Teflon_VideoProcess', f'frames_Process_{fps}')
        ffmpeg_frame_extractor(
            video_path,
            output_frame_pattern=frame_path,
            fps=fps,
            wipe=True,
            use_cuda=False,
            grayscale=True,
            health_check=True
        )
    
    # vv = get_subdirectories(BaseAddress, max_depth=7)
    # print(f"Found {len(vv)} subdirectories in {BaseAddress}")