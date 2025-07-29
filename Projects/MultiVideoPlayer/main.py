import  os
import  cv2
import  glob
import  numpy as np

import time
"""
Fix:    
    Fixing name of video
    Returning when all videos finished

"""

def ClenaUp(caps):
    # Cleanup
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

def MultiVideo(video_paths:list, VideoGrid = (3, 5), output_size = (400, 400),
               paused = False, show_paths = False, show_progress = False,)->bool:
    
    _row, _col      = VideoGrid
    num_videos      = _row * _col
    # Initialize video capture objects
    caps            = []
    total_frames    = []
    video_labels    = []

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
        video_labels.append(f'{path.split("/")[6]}-{path.split("/")[7]}-{path.split("/")[8][28:30]}*{path.split("/")[8][42:]}')  # Precompute labels

    frame_shape = (output_size[1], output_size[0], 3)
    blank_frame = np.zeros(frame_shape, dtype=np.uint8)
    frames      = [blank_frame.copy() for _ in range(num_videos)]

    # GUI state
    window_name = f'{_row}x{_col} Video Grid'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
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
            break

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

    ClenaUp(caps)
    return 0

if __name__ == "__main__":
    # Load videos
    videos = glob.glob("/media/d2u25/Dont/S4S-ROF/frame_Extracted/*/*/*/SR_edge/*.mp4")
    videos.sort(key=lambda x: x[1], reverse=True)
    
    _end = len(videos)
    for lis in range(len(videos)-15,0,-15):
        print(len(videos[lis:_end]))
        MultiVideo(videos[lis:_end])
        _end = lis
        time.sleep(20)