import cv2
import os
import glob
import tqdm
# Get list of .mp4 files
videos = glob.glob("/media/d2u25/Dont/S4S-ROF/frame_Extracted/*/*/*/SR_edge/*.mp4")

healthy_videos = []

for video in tqdm.tqdm(videos):
    cap = cv2.VideoCapture(video)
    ret, _ = cap.read()
    cap.release()
    
    if not ret:
        print(f"Corrupt or unreadable video: {video}")
        os.remove(video)
    else:
        file_size = os.path.getsize(video)
        healthy_videos.append((video, file_size))

# Sort by file size (descending)
healthy_videos.sort(key=lambda x: x[1], reverse=True)

print("\nHealthy videos sorted by file size:")
for path, size in healthy_videos:
    print(f"{path} ({size / (1024*1024):.2f} MB)")
