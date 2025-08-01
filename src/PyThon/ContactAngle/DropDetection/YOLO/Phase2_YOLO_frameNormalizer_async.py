import  os
import  cv2
import  tqdm
import  natsort

import  numpy               as      np
import  matplotlib.pyplot   as      plt

from    ultralytics         import  YOLO
from    concurrent.futures  import  ThreadPoolExecutor

def load_files(ad):
    valid_extensions = {"tiff", "tif", "png", "jpg", "jpeg", "bmp", "gif", "webp"}  # Common image formats
    FileNames = []
    for file in sorted(os.listdir(ad)):
        try:
            if file.split(".")[-1].lower() in valid_extensions:
                FileNames.append(file)
        except IndexError:
            pass
    return natsort.natsorted(FileNames)

def get_subdirectories(root_dir, max_depth=2):
    directories = []
    for root, dirs, _ in sorted(os.walk(root_dir)):
        if root == root_dir:
            continue  # Skip the root directory itself
        depth = root[len(root_dir):].count(os.sep)
        if depth < max_depth:
            directories.append(root)
        else:
            del dirs[:]  # Stop descending further
    return directories

def process_image(file_address):
    """Process a single image."""
    image = cv2.imread(file_address, cv2.IMREAD_UNCHANGED)  # Faster loading
    if image is None:
        return  # Skip unreadable images

    # Run YOLO inference
    results = yolo_m.predict(image, verbose=False, device="cuda")  # Use GPU if available

    for res in results:
        boxes = res.boxes.xyxy  # Convert tensor to NumPy  .cpu().numpy().astype(np.float32)
        if boxes.size == 0:  # Skip empty detections
            # os.remove(file_address)
            return

        x1, _, x2, _ = boxes[0]  # Extract bounding box

        # if not (40 < x1 < 1205):  # Check filtering condition
        if x2 < 1200 and 40 < x1:
            # os.remove(file_address)
            pass

def process_experiment(experiment):
    """Process all images in an experiment using multithreading."""
    file_addresses = [os.path.join(experiment, f) for f in load_files(experiment)]
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust based on CPU cores
        list(tqdm.tqdm(executor.map(process_image, file_addresses), total=len(file_addresses)))


if __name__=="__name__":
    # import matplotlib
    # matplotlib.use('TkAgg')

    # Load model once
    yolo_m = YOLO("models/best.pt")


    # Iterate over experiments
    for tilt in get_subdirectories(r"Bubble")[0]:
        for experiment in get_subdirectories(tilt):
            process_experiment(experiment)