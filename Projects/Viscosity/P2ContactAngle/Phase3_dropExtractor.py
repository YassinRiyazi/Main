import glob
import os
import cv2
import multiprocessing
import pandas as pd
from tqdm import tqdm
import subprocess
import numpy                as np
import matplotlib.pyplot    as plt
from PyThon.ContactAngle.DropDetection.DropDetection_YOLO import process_experiment

def walk_forward(array, steep = 0.0025)->int:
    """
    Walk forward from the start index until we find a value greater than 0.

    Steep: a parameter that defines the steepness of the slope to detect the drop.
    """
    for i in range(2, len(array)-10, 1):
        if np.abs(array[i] - array[i-1])> steep:
            return i - 1
    raise ValueError("No drop detected in the image. Please check the image or parameters.")
    return None

def backward(array, steep = 0.0025)->int:
    """
    Walk forward from the start index until we find a value greater than 0.

    Steep: a parameter that defines the steepness of the slope to detect the drop.
    if want to include wider dropes with low slope decrese the sensitivity of the slope.
    For example, if you want to detect drops with low surface tention, surfactant, slope shold be less than 0.0025
    """
    for i in range(len(array)-10, 10, -1):
        if np.abs(array[i] - array[i-1])> steep:
            return i
        
    raise ValueError("No drop detected in the image. Please check the image or parameters.")
    return None
       
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
def detect_drop(image,dims,show = False,scaleDownFactorx = 5, scaleDownFactory =2)-> np.ndarray:
    """
    Trying to detect images with less than 2ms delay.

    [V] 0. Convert image to grayscale,
    [V] 1. Resizing image,
    [V] 2. Applyng gussina blur (morphologyEx worked better than Gaussian blur),
    [-] 3. Transpoing image (Optimization puposes) [Didn't improve any thing and because of if damaged was more than benefits],
    [V] 4. Summation over rows
    [-] 5. Normalize images based on height and maximum brightness,
    [V] 6. Finding beggining of the drop and ending of the drop,
    and finally drawing a rectangle around the drop.


    Caution:
        Code will fail if there are more than one drop in the image.
        Images should have exactly 5 rows of black pixels at the bottom of the image.
        It works with tilted setup drop imaeges, on other drop shape it is untested


    Note:
        Used morphologyEx desipte its been applied to frames. Some time in the end of slide, small drops
        cause immature termination of the drop detection, then we would have a very long drop.

        Since chance of having a 

    """
    resized_image = image
    # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Resize the image to a smaller size for faster processing
    resized_image = cv2.resize(resized_image, (dims[1]//scaleDownFactorx, dims[0]//scaleDownFactory))
    vv = resized_image.sum(axis=0)
    # vv = vv/(dims[0]-15)/scaleDownFactory  # Normalize the sum of rows
    vv = vv/vv.mean()  # Normalize the sum of rows to have a mean of 1

    
    resized_image = cv2.morphologyEx(resized_image, cv2.MORPH_CLOSE, kernel)

    ## Close operation fills small dark holes # Kernel size depends on spot size
    vv_mor = resized_image.sum(axis=0)
    vv_mor = vv_mor/vv_mor.mean()  # Normalize the sum of rows to have a mean of 1

    if show:
        print("Sum of rows:", vv.shape, "image shape", resized_image.shape)
        resized_image = cv2.flip(resized_image, 1) # not neccessary
        print("Sum of rows:", vv.shape)
        cv2.imshow("Resized Image", resized_image)
        
        plt.plot(vv)
        plt.title("Sum of Rows")
        plt.xlabel("Column Index")
        plt.ylabel("Sum Value")
        plt.show()

    return vv, vv_mor

def bottom_row_unifier(image, base, height, target_height=170, bottom_padding=5):
    """
    Unifies the bottom rows of an image to a specified target height.
    
    :param image_path: Path to the input image.
    :param target_height: Desired height of the output image.
    :return: Processed image with unified bottom rows.
    """
    # ## Step 1: Resize the image if necessary
    # resized_image   = image[:,::50]
    # vv              = resized_image.sum(axis=1)
    # height = len(vv)
    # """
    #     Resizing is mistake. 
    #     Do the summation in loop and stop when sum is more than one
    # """
    # for i in range(height-1, 0, -1):
    #     if vv[i] > 0:
    #         i -= 1
    #         print(f"Bottom row found at index {i}, height: {height}")
    #         break
    
 
    padding_top = target_height - height
    image = cv2.copyMakeBorder(image[:base,:], padding_top, 0, 0, 0, cv2.BORDER_CONSTANT, None, value = 255)

    # image = cv2.copyMakeBorder(image[:,:], 0, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)

    return image

def bottom_row_unifier_all(image):
    resized_image   = image[:,::50]
    vv              = resized_image.sum(axis=1)
    for i in range(len(vv)-1, 0, -1):
        if vv[i] > 0:
            i -= 1
            return i




def crop_and_save_image(image, output_path, x1, x2,tolerance=3 ):
    """
    Crops a region from the input image and saves it to the output path.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the cropped image.
        x1 (int): Top-left x-coordinate.
        x2 (int): Bottom-right x-coordinate.

    Raises:
        ValueError: If the crop coordinates are invalid or image cannot be loaded.

    Example:
        crop_and_save_image("input.jpg", "output.jpg", 10, 20, 100, 200)
    """

    if x1 >= x2:
        raise ValueError("Invalid crop coordinates: x1 >= x2 or y1 >= y2")

    cropped = image[:, x1-tolerance:x2+tolerance]
    cropped = cv2.bitwise_not(cropped)
    cv2.imwrite(output_path, cropped)
    

def Main(experiment):
    """
    Crops all images for a single experiment folder and saves crop info to CSV.
    """
    images = sorted(glob.glob(os.path.join(experiment, '*.jpg')))
    if not images:
        return

    desfolder_crop = images[0].replace('frames', 'frame_Extracted')[:-11]
    os.makedirs(desfolder_crop, exist_ok=True)

    desfolder_csv = images[0].replace('frames', 'frame_Extracted_xx')[:-11]
    os.makedirs(desfolder_csv, exist_ok=True)
    
    scaleDownFactorx = 5
    rows = []

    drop_width = 300

    try:
        for image in images:
            frame     = cv2.imread(image)
            frame     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            vv, vv_mor  = detect_drop(frame, frame.shape, show=False, scaleDownFactorx=scaleDownFactorx)
            endpoint    = walk_forward(vv_mor)
            beginning   = backward(vv)

            endpoint  = int(endpoint * scaleDownFactorx)
            beginning = int(beginning * scaleDownFactorx) 

            if (beginning - endpoint > drop_width):
                print(image)
                endpoint  = walk_forward(vv_mor, steep=0.005)
                # beginning = backward(vv)
                endpoint  = int(endpoint * scaleDownFactorx)
                # beginning = int(beginning * scaleDownFactorx) 

            save_path = image.replace('frames', 'frame_Extracted')
            crop_and_save_image(frame, save_path, endpoint, beginning)

            img_name = os.path.basename(image)
            rows.append({'image': img_name, 'endpoint': endpoint, 'beginning': beginning})

            drop_width = int (1.15 * (beginning - endpoint))

        # Save results as a single CSV file
        csv_path = os.path.join(desfolder_csv, 'detections.csv')
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    except Exception as e:
        print(f"Error processing {experiment}: {e}")
        # Clean up folders on failure
        _temp = experiment.replace('frames', 'frame_Extracted')
        subprocess.run(["rm", "-rf", _temp])
        _temp = experiment.replace('frames', 'frame_Extracted_xx')
        subprocess.run(["rm", "-rf", _temp])


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


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    
    # Collect all experiment paths
    root_folder_name = "/media/d2u25/Dont/S4S-ROF/frames"
    experiment_paths = []
    for tilt in get_subdirectories(root_folder_name):
        for fluid in get_subdirectories(tilt):
            experiment_paths.extend(get_subdirectories(fluid))
    print(f"Found {len(experiment_paths)} experiments.")

    # Use tqdm to track progress
    with multiprocessing.Pool(processes=min(12, os.cpu_count())) as pool:
        list(tqdm(pool.imap_unordered(Main, experiment_paths), total=len(experiment_paths)))
                
    # adress = r"/media/d2u25/Dont/S4S-ROF/frames/300/S3-SNr3.05_D/frames20250707_201500_DropNumber_11"
    # Main(adress)
    """
        Check the YOLO result with OpenCV vcountors
        Normalize the white lines in bottom of the images
        save x1 in the textfile with same name as the image
    """