import  os
import  cv2
import  tqdm
import  numpy               as      np
import  matplotlib.pyplot   as      plt
from    ultralytics         import  YOLO

import multiprocessing


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

def load_files(ad):
    valid_extensions = {"tiff", "tif", "png", "jpg", "jpeg", "bmp", "gif", "webp"}  # Common image formats
    FileNames = []
    for file in sorted(os.listdir(ad)):
        try:
            if file.split(".")[-1].lower() in valid_extensions:
                FileNames.append(file)
        except IndexError:
            pass
    return sorted(FileNames)

def _forward(experiment,model):
    for i in (load_files(experiment)):
        file_adress = os.path.join(experiment,i)
        image       = cv2.imread(file_adress)
        
        # Perform batch YOLO prediction
        results = model(image, verbose=False)

        for file_idx, res in enumerate(results):
            if res.boxes.xyxy.shape[0]==0:
                os.remove(file_adress)
                continue
            x1, _, x2, _ = np.array(res.boxes.xyxy[:, :].cpu().numpy(), dtype=np.float32)[0]

            if x2 < 1245-40 and 40 < x1:
                return None
            else:
                os.remove(file_adress)

def _backward(experiment,model):
    for i in (reversed(load_files(experiment))):
        file_adress = os.path.join(experiment,i)
        image       = cv2.imread(file_adress)
        
        # Perform batch YOLO prediction
        results = model(image, verbose=False)

        for file_idx, res in enumerate(results):
            if res.boxes.xyxy.shape[0]==0:
                # print(f"No drop detected, probably out of scope. {file_adress}")
                os.remove(file_adress)
                continue
            x1, _, x2, _ = np.array(res.boxes.xyxy[:, :].cpu().numpy(), dtype=np.float32)[0]

            if x2 < 1245-40 and 40 < x1:
                return None
            else:
                os.remove(file_adress)

def process_experiment(experiment):
    yolo_m = YOLO("models/best.pt")
    _forward(experiment,yolo_m.predict)
    _backward(experiment,yolo_m.predict)

if __name__ == "__main__":
    # import matplotlib
    # matplotlib.use('TkAgg')

    # Load the model
    # yolo_m = YOLO("models/best.pt")
    import glob
    import utils
    experiments = []
    for tilt in utils.get_subdirectories(r"frames"):
        for fluid in utils.get_subdirectories(tilt):
            for video in utils.get_subdirectories(fluid):
                experiments.append(video)

    # Use multiprocessing pool
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()//3) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_experiment, experiments), total=len(experiments)))




    # for tilt in get_subdirectories(r"Bubble"):
    #     for experiment in tqdm.tqdm(get_subdirectories(tilt)):
    #         _forward(experiment,yolo_m.predict)
    #         _backward(experiment,yolo_m.predict)


"""
Change log

V2.0.0
    Added Multiprocess

    GPU Utils:96%
    CPU Utils:50%
    5H-> 19M
V1.0.0
    Initiated
    GPU Utils:40%
    CPU Utils:20%
"""