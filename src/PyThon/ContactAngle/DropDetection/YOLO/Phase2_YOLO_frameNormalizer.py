import  os
import  cv2
import  tqdm

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
    return sorted(FileNames)

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

def singleFolderDropNormalizer(images: list[os.PathLike]):
    """
    This function normalizes the drop images in a single folder.
    It removes images that do not meet the criteria defined by the YOLO model.
    """
    # Load the model
    BaseAddress = os.path.dirname(__file__)
    weights_path = os.path.join(BaseAddress,'..', "Weights", "Gray-320-s.engine")
    weights_path = os.path.normpath(weights_path)
    yolo_m = YOLO(weights_path, task="detect",verbose=False)

    def _forward(image):
        """
        Forward pass through the YOLO model.
        Returns the bounding box coordinates.
        """
        for image in tqdm.tqdm(images):
            image       = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            image       = cv2.resize(image, (640, 640))  # Resize to match YOLO input size
            image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            results     = yolo_m.predict(image, verbose=False)

            for file_idx, res in enumerate(results):
                x1, _, x2, _ = np.array(res.boxes.xyxy[:, :].cpu().numpy(), dtype=np.float32)[0]

                if x2 < 1200:
                    return True
                else:
                    os.remove(image)
                    break  # Exit after removing the first invalid image
    def _backward(images):
        """
        Backward pass through the YOLO model.
        This function is not used in this context but is kept for consistency.
        """
        for image in reversed(images):
            image       = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            image       = cv2.resize(image, (640, 640))  # Resize to match YOLO input size
            image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            results     = yolo_m.predict(image, verbose=False)

            for file_idx, res in enumerate(results):
                x1, _, x2, _ = np.array(res.boxes.xyxy[:, :].cpu().numpy(), dtype=np.float32)[0]

                if x1 < 40:
                    return True
                else:
                    os.remove(image)
                    break  # Exit after removing the first invalid image
    _forward(images)
    _backward(images)


if __name__ == "__main__":
    # import matplotlib
    # matplotlib.use('TkAgg')

    BaseAddress = os.path.dirname(__file__)
    weights_path = os.path.join(BaseAddress,'..', "Weights", "Gray-320-s.engine")
    weights_path = os.path.normpath(weights_path)
    yolo_m = YOLO(weights_path, task="detect",verbose=False)

    import glob
    images = glob.glob(os.path.join('/media/d2u25/Dont/frames/280/S3-SNr3.01_D/T111_01', '*.png'))
    singleFolderDropNormalizer(images)