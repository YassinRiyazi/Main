import cv2
import os
import tqdm
import utils
import numpy as np
from ultralytics import YOLO

def count_files(directory):
    return sum(len(files) for _, _, files in os.walk(directory))

def crop_white_bottom(image, threshold=250):
    """Crops white rows from the bottom of the image based on a pixel threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row_thresh = np.all(gray > threshold, axis=1)

    # Find the first row from the bottom that is NOT fully white
    cut_index = len(row_thresh)
    for i in reversed(range(len(row_thresh))):
        if not row_thresh[i]:
            cut_index = i + 1
            break

    return image[:cut_index]


# if __name__ == "__main__":
#     import pandas as pd
#     root_folder_name = "frames"
#     yolo_m  = YOLO("Weights/Gray-320-s.engine",
#                    task='detect',
#                   verbose=False)

#     batch_size = 16  # adjust this depending on your GPU memory

#     for tilt in utils.get_subdirectories(root_folder_name):
#         for fluid in utils.get_subdirectories(tilt):
#             for experiment in tqdm.tqdm(utils.get_subdirectories(fluid)):

#                 _loc_path = os.path.join(experiment, 'drops')
#                 processed_flag = os.path.join(_loc_path, ".processed_YOLO")

#                 if os.path.isfile(processed_flag):
#                     continue

#                 if not os.path.isdir(_loc_path):
#                     os.mkdir(_loc_path)

#                 name_files = utils.load_files(experiment)
#                 img_paths = [os.path.join(experiment, name_files[i]) for i in range(1, len(name_files))]

#                 all_detections = []  # store detection info

#                 # Process in batches
#                 for i in range(0, len(img_paths), batch_size):
#                     batch_paths = img_paths[i:i + batch_size]
#                     batch_imgs = [cv2.imread(p) for p in batch_paths]

#                     results = yolo_m(batch_imgs, verbose=False)

#                     for res, path in zip(results, batch_paths):
#                         if len(res.boxes) > 0:
#                             for box in res.boxes.xyxy.cpu().numpy():
#                                 x1, y1, x2, y2 = map(int, box)
#                                 all_detections.append({
#                                     "image": os.path.basename(path),
#                                     "x1": x1,
#                                     "y1": y1,
#                                     "x2": x2,
#                                     "y2": y2
#                                 })

#                 # Save to CSV
#                 if all_detections:
#                     df = pd.DataFrame(all_detections)
#                     csv_path = os.path.join(_loc_path, "detections.csv")
#                     df.to_csv(csv_path, index=False)

#                 utils.writter(processed_flag)


import os
import cv2
import tqdm
import numpy as np
import pandas as pd
from ultralytics import YOLO
import utils
import multiprocessing

# Global variable
yolo_m = None

def init_worker(yolo_weights_path):
    """
    Initializes YOLO model in the worker process.
    This will be called once per process.
    """
    global yolo_m
    yolo_m = YOLO(yolo_weights_path, task='detect', verbose=False)

def process_fluid(fluid_path):
    """
    Processes all experiments inside a fluid folder using YOLO object detection.
    Saves detection results as CSVs inside each experiment.
    """
    global yolo_m
    assert yolo_m is not None, "YOLO model not initialized in this worker!"

    for experiment in utils.get_subdirectories(fluid_path):
        _loc_path = os.path.join(experiment, 'drops')
        processed_flag = os.path.join(_loc_path, ".processed_YOLO")

        if os.path.isfile(processed_flag):
            continue

        if not os.path.isdir(_loc_path):
            os.mkdir(_loc_path)

        name_files = utils.load_files(experiment)
        img_paths = [os.path.join(experiment, name_files[i]) for i in range(1, len(name_files))]

        all_detections = []

        for img_path in img_paths:
            img = cv2.imread(img_path)
            results = yolo_m(img, verbose=False)
            res = results[0]

            if len(res.boxes) > 0:
                for box in res.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    all_detections.append({
                        "image": os.path.basename(img_path),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

        if all_detections:
            df = pd.DataFrame(all_detections)
            csv_path = os.path.join(_loc_path, "detections.csv")
            df.to_csv(csv_path, index=False)

        utils.writter(processed_flag)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)  # needed for CUDA with multiprocessing

    root_folder_name = "/media/d2u25/Dont/S4S-ROF/frames"
    yolo_weights_path = "/home/d2u25/Desktop/Main/Projects/ContactAngle/Weights/Gray-320-s.engine"

    fluid_dirs = []
    for tilt in utils.get_subdirectories(root_folder_name):
        fluid_dirs.extend(utils.get_subdirectories(tilt))

    print(f"Found {len(fluid_dirs)} fluids to process")

    # Adjust number of workers based on system resources
    num_workers = min(2, len(fluid_dirs))

    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(yolo_weights_path,)
    ) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_fluid, fluid_dirs), total=len(fluid_dirs)))
