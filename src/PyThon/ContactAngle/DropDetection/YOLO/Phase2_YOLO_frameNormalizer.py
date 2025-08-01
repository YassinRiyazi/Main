import  os
import  cv2
import  tqdm
import  natsort

import  numpy               as      np
import  matplotlib.pyplot   as      plt

from    ultralytics         import  YOLO

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


if __name__=="__name__":
    # import matplotlib
    # matplotlib.use('TkAgg')

    # Load the model
    yolo_m = YOLO("models/best.pt")

    for tilt in get_subdirectories(r"Bubble")[0]:
        for experiment in get_subdirectories(tilt):
            for i in tqdm.tqdm(load_files(experiment)):
                print(i)
                file_adress = os.path.join(experiment,i)
                image       = cv2.imread(file_adress, cv2.IMREAD_UNCHANGED)
                results     = yolo_m.predict(image, verbose=False)

                for file_idx, res in enumerate(results):
                    x1, _, x2, _ = np.array(res.boxes.xyxy[:, :].cpu().numpy(), dtype=np.float32)[0]

                    if x2 < 1200 and 40 < x1:
                        pass
                    else:
                        os.remove(file_adress)