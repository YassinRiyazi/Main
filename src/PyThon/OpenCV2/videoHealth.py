"""
    # -*- coding: utf-8 -*-
    Author: Yassin Riyazi
    Date: 03-07-2025

    Checking health of all videos.
"""
import cv2
import os
import glob
from typing import Union, Optional, Tuple
import tqdm


def check_videos(video_addresses: list[Union[os.PathLike, str]],
                 remove_corrupted: bool = False,
                 verbose: bool = False) -> Tuple[list[str],list[Tuple[str, int]], list[str]]:
    """
    Check all videos matching a glob pattern, remove corrupted ones, and return healthy videos sorted by size.
    
    Args:
        video_addresses (list[union[os.PathLike, str]]): List of video file paths to check.

    Returns:
        list[tuple[str, int]]: List of tuples (video_path, file_size_bytes) sorted by file size descending.
        list[str]: List of corrupted video paths.
    ```
    print("Healthy videos sorted by file size:")
    for path, size in healthy_videos:
        print(f"{path} ({size / (1024*1024):.2f} MB)")
    ```
    """
    _local_video_addresses = []
    healthy_videos = []
    broken_videos = []

    for video in tqdm.tqdm(video_addresses, desc="Checking videos"):
        cap = cv2.VideoCapture(video)
        ret, _ = cap.read()
        cap.release()

        if not ret:
            print(f"Corrupt or unreadable video: {video}")
            broken_videos.append(video)
            if verbose:
                print(f"Removing corrupted video: {video}")
            if remove_corrupted:
                os.remove(video)
        else:
            _local_video_addresses.append(video)
            file_size = os.path.getsize(video)
            healthy_videos.append((video, file_size))

    healthy_videos.sort(key=lambda x: x[1], reverse=True)

    return _local_video_addresses, healthy_videos, broken_videos


if __name__ == "__main__":
    # Example usage
    video_pattern = "/media/d2u25/Dont/Teflon_VideoProcess/*/*/*/*.mp4"
    video_files = glob.glob(video_pattern)

    healthy_videos, broken_videos = check_videos(video_files, remove_corrupted=False, verbose=True)
    print(f"Total healthy videos: {len(healthy_videos)}")
    print(f"Total broken videos: {len(broken_videos)}")
