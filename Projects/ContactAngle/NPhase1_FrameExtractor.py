"""
    # -*- coding: utf-8 -*-
    Author: Yassin Riyazi
    Date: 01-07-2025
"""
import os
import sys

# Add the absolute path to the ./src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', 'src')))
from PyThon.FFMpeg import ffmpeg_frame_extractor

if __name__ == "__main__":
    video_path = "/media/d2u25/Dont/S4S-ROF/Teflon/280/S2-SNr2.9_D/frames20250621_194738_DropNumber_01.mp4"

    _temp = os.path.split(video_path)
    _folderName = f"T{_temp[1][:-4][18:21]}_{_temp[1][-6:-4]}"
    frame_path = os.path.join(_temp[0], _folderName,  'frame_%06d.png')

    ffmpeg_frame_extractor(
        video_path,
        output_frame_pattern=frame_path,
        fps=30,
        wipe=True,
        use_cuda=False,
        grayscale=True,
        health_check=True
    )