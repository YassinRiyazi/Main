import cv2
import numpy as np

# List of video file paths
video_paths = [
    'Projects/MultiVideoPlayer/result.mp4',
    'Projects/MultiVideoPlayer/result.mp4',
    'Projects/MultiVideoPlayer/result.mp4',
    'Projects/MultiVideoPlayer/result.mp4',
    'Projects/MultiVideoPlayer/result.mp4',
    'Projects/MultiVideoPlayer/result.mp4',
    'Projects/MultiVideoPlayer/result.mp4',
    'Projects/MultiVideoPlayer/result.mp4',
    'Projects/MultiVideoPlayer/result.mp4',
]

# Initialize video capture objects with hardware acceleration
caps = []
for path in video_paths:
    # Used cv2.CAP_FFMPEG to leverage FFmpeg backend
    # which can utilize hardware acceleration (e.g., GPU decoding) 
    # if supported by the system. This reduces , CPU load for video decoding.
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG) 
    if not cap.isOpened():
        print(f"Error: Could not open video {path}")
        for c in caps:
            c.release()
        exit()
    caps.append(cap)

# Define output size for each video
output_size = (400, 400)
grid_size = (output_size[1] * 3, output_size[0] * 3, 3)  # 1200x1200x3 for RGB

# Pre-allocate grid array
grid = np.empty(grid_size, dtype=np.uint8)

# Create window with optimized flags
cv2.namedWindow('3x3 Video Grid', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

# Pre-allocate frame buffers
frames = [np.empty((output_size[1], output_size[0], 3), dtype=np.uint8) for _ in range(9)]

while True:
    all_valid = True

    # Read and resize frames
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            all_valid = False
            break
        # Resize with INTER_LINEAR for speed (faster than default INTER_LINEAR)
        frames[i] = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR) # cv2.INTER_LINEAR interpolation, which is faster than the default

    if not all_valid:
        break

    # Concatenate frames into 3x3 grid using array slicing
    grid[0:output_size[1], 0:output_size[0]] = frames[0]
    grid[0:output_size[1], output_size[0]:2*output_size[0]] = frames[1]
    grid[0:output_size[1], 2*output_size[0]:3*output_size[0]] = frames[2]
    grid[output_size[1]:2*output_size[1], 0:output_size[0]] = frames[3]
    grid[output_size[1]:2*output_size[1], output_size[0]:2*output_size[0]] = frames[4]
    grid[output_size[1]:2*output_size[1], 2*output_size[0]:3*output_size[0]] = frames[5]
    grid[2*output_size[1]:3*output_size[1], 0:output_size[0]] = frames[6]
    grid[2*output_size[1]:3*output_size[1], output_size[0]:2*output_size[0]] = frames[7]
    grid[2*output_size[1]:3*output_size[1], 2*output_size[0]:3*output_size[0]] = frames[8]

    # Display the grid
    cv2.imshow('3x3 Video Grid', grid)

    # Adjust waitKey for ~30 fps (assuming standard video frame rate)
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

# Release resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()