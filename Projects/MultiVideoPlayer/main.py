"""
    Author:      Yassin Riyazi
    Date:        29-07-2025
    Description: This script opens multiple videos in a grid format, allowing for easy viewing and navigation.
"""
import  os
import  cv2
import  time
import  glob
import  numpy as np


def CleanUp(caps: list[cv2.VideoCapture]) -> None:
    """
    Releases all video capture objects and closes all OpenCV windows.
    args:
        caps (list[cv2.VideoCapture]): List of video capture objects to release.
    returns:
        None
    """
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

def MultiVideo(video_paths: list, VideoGrid: tuple = (3, 5), output_size=(400, 400),
               paused: bool = False, show_paths: bool = False, show_progress: bool = False) -> bool:
    """
    Opens multiple videos in a grid format.
    args:
        video_paths (list): List of paths to video files.
        VideoGrid (tuple): Tuple defining the grid size (rows, columns).
        output_size (tuple): Size to which each video frame will be resized.
        paused (bool): Whether the video playback starts paused.
        show_paths (bool): Whether to display video paths on the frames.
        show_progress (bool): Whether to show progress percentage on the frames.
        
    returns:
        bool: True if successful, False if no videos are provided.

    TODO:    
        [V] Fixing name of video
        [V] Returning when all videos finished
        [V] Adding a pause function
        [V] Adding a toggle for showing video paths
        [V] Adding a toggle for showing progress percentage
        [V] Adding a toggle for full screen
        [ ] Making it C++ with OpenCV and OpenGL with CUDA support
    """
    _row, _col = VideoGrid
    num_videos = _row * _col
    # Initialize video capture objects
    caps = []
    total_frames = []
    video_labels = []

    for path in video_paths:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            for c in caps:
                c.release()
            exit()
        caps.append(cap)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames.append(total)
        video_labels.append(f'{path.split("/")[-4]}/{path.split("/")[-3]}/{path.split("/")[-2]}')  # Precompute labels

    frame_shape = (output_size[1], output_size[0], 3)
    blank_frame = np.zeros(frame_shape, dtype=np.uint8)
    frames = [blank_frame.copy() for _ in range(num_videos)]

    # GUI state
    window_name = f'{_row}x{_col} Video Grid'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # Set window to full screen
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Set window to stay on top
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        if not paused:
            checking_end_all_video = 0
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret or frame is None:
                    frames[i][:] = 0
                    continue

                # Resize if needed
                if frame.shape[1::-1] != output_size:
                    frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)

                if show_paths:
                    cv2.putText(frame, video_labels[i], (10, 30), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                pos     = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                total   = total_frames[i]
                percent_left = 100 - int((pos / total) * 100)
                checking_end_all_video += percent_left
                if show_progress: 
                    text = f"{percent_left}% left"
                    cv2.putText(frame, text, (10, 55), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                frames[i] = frame
        
            if checking_end_all_video == 0:
                break

        # Build grid efficiently
        row_stack   = [np.hstack(frames[r*_col:(r+1)*_col]) for r in range(_row)]
        grid        = np.vstack(row_stack)

        # Show the grid
        cv2.imshow(window_name, grid)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            CleanUp(caps)
            return 0

        elif key == ord(' '):
            paused = not paused

        elif key in (ord('a'), ord('A')):
            show_paths      = not show_paths

        elif key in (ord('v'), ord('V')):
            show_progress   = not show_progress

        elif key == 81:  # Left arrow
            if paused:
                for i, cap in enumerate(caps):
                    cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(cur - 30, 0))

        elif key == 83:  # Right arrow
            if paused:
                for i, cap in enumerate(caps):
                    cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, min(cur + 30, total_frames[i]))
        
        elif key == ord('f'):  # Toggle full screen with 'f' key
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    CleanUp(caps)
    return 0

if __name__ == "__main__":
    # Load videos
    videos = []
    for tilt in glob.glob("/media/d2u25/Dont/Teflon_VideoProcess/*"):
        for experiment in glob.glob(os.path.join(tilt,'*')):
            for _idx, rep in enumerate(glob.glob(os.path.join(experiment,'*','result.mp4'))):
                if _idx < 1:
                    videos.append(rep)


    # videos.sort(key=lambda x: x[1], reverse=True)
    
    _end = len(videos)
    for lis in range(len(videos)-15,0,-15):
        print(len(videos[lis:_end]))
        MultiVideo(videos[lis:_end],show_paths=True, show_progress=True)
        _end = lis
