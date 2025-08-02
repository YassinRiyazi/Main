from multiprocessing import pool
import  os
import  cv2
import  shutil
import  tqdm
import  glob
import  subprocess
import  multiprocessing
 
import  numpy               as      np
import matplotlib
matplotlib.use('Agg')  # For file output only, no GUI
import matplotlib.pyplot as plt
from    skimage.measure     import  ransac, LineModelND

import sys
# Add the absolute path to the ./src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', 'FFMpeg')))
from Jpg2Video import create_video_from_images

def fit_and_rotate_image(image_path: os.PathLike,
                         experiment: str = None,
                         results: bool = True,
                         focus_ratio: float = 0.3) -> tuple[float, tuple, np.ndarray]:
    """
    Fits a robust line to the bottom edges of an image and rotates the image to level the surface.
    
    Args:
        image_path (os.PathLike): Path to the input image.
        experiment (str, optional): Experiment name for saving results.
        results (bool, optional): If True, saves a diagnostic plot.
        focus_ratio (float, optional): Portion of the image height to analyze from the bottom. Default 0.3.
    
    Returns:
        tuple:
            - angle (float): Rotation angle in degrees.
            - image_shape (tuple): Original image shape.
            - rotated_image (np.ndarray): Rotated image.

    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/BaseLine/doc/result.png" alt="Italian Trulli">

    """
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # # loc_image = image[:, :-410]
    
    # # Detect edges using Canny
    # edges = cv2.Canny(image, 50, 150)
    
    # # Get edge coordinates
    # y_indices, x_indices = np.where(edges > 0)
    # points = np.column_stack((x_indices, y_indices))
    
    # # Apply RANSAC to fit a line
    # model, inliers = ransac(points, LineModelND, min_samples=2, residual_threshold=2, max_trials=1000)
    
    # # Get line parameters
    # line_x = np.array([min(x_indices), max(x_indices)])
    # line_y = model.predict_y(line_x)
    
    # # Compute angle of rotation
    # dx = line_x[1] - line_x[0]
    # dy = line_y[1] - line_y[0]
    # angle = np.degrees(np.arctan2(dy, dx))
    # # Rotate image to level the line
    # (h, w) = image.shape[:2]
    # center = (w // 2, h // 2)
    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    # Load grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]

    # Focus on the bottom region
    focus_height = int(h * focus_ratio)
    bottom_region = image[h - focus_height:h, :]

    # Preprocess to stabilize edges
    blurred = cv2.GaussianBlur(bottom_region, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Get edge coordinates
    y_indices, x_indices = np.where(edges > 0)
    if len(x_indices) < 2:
        raise ValueError(f"Not enough edge points detected in {image_path}")
    points = np.column_stack((x_indices, y_indices))

    # Fit robust line using RANSAC
    model, inliers = ransac(
        points, LineModelND,
        min_samples=2,
        residual_threshold=1.0,  # tighter fit
        max_trials=5000          # more attempts
    )

    # Compute line endpoints
    line_x = np.array([min(x_indices), max(x_indices)])
    line_y = model.predict_y(line_x)

    # Adjust for cropped region
    line_y += (h - focus_height)

    # Compute angle
    dx = line_x[1] - line_x[0]
    dy = line_y[1] - line_y[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate around bottom-center to preserve surface alignment
    center = (w // 2, h - 1)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=255)
    # Optional visualization
    if results:
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.plot(line_x, line_y, color='red', linewidth=2)
        plt.title("Detected Line")
        plt.subplot(1, 2, 2)
        plt.imshow(rotated_image, cmap='gray')
        plt.title("Rotated Image")
        save_dir = os.path.join(os.path.dirname(image_path), "rotation")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "result.png"), dpi=300)
        plt.close()

    return angle, image.shape, rotated_image

def fit_image(image: cv2.Mat, black_base_line=10):
    """
    Fits a line to the detected edges in a grayscale image using RANSAC, and computes 
    the vertical offset from the fitted line to a given black baseline.

    Args:
        image (cv2.Mat): Grayscale input image.
        black_base_line (int, optional): Reference baseline offset in pixels. Defaults to 10.

    Returns:
        int: Vertical offset (in pixels) between the fitted line's center height and the black baseline.
    """
    # Detect edges using Canny edge detector
    edges = cv2.Canny(image, 50, 150)
    
    # Find coordinates of non-zero (edge) pixels
    y_indices, x_indices = np.where(edges > 0)
    points = np.column_stack((x_indices, y_indices))  # Shape: (N, 2)

    # Fit a robust line to the edge points using RANSAC (to handle outliers)
    model, inliers = ransac(points, LineModelND, min_samples=2,
                            residual_threshold=2, max_trials=1000)
    
    # Define X-range of the line (min to max X in the edge points)
    line_x = np.array([min(x_indices), max(x_indices)])
    
    # Predict corresponding Y values from the fitted line model
    line_y = model.predict_y(line_x)
    
    # Compute angle of the line (not used in return, but may be useful for debugging)
    dx = line_x[1] - line_x[0]
    dy = line_y[1] - line_y[0]
    angle = np.degrees(np.arctan2(dy, dx))  # Angle of the fitted line in degrees

    # Compute average height of the line and subtract the baseline
    return int((line_y[1] + line_y[0]) // 2) - black_base_line

def line_finder(image:cv2.Mat, rotation_matrix:cv2.Mat, black_base_line:int = 10) -> int:
    """
    Finds the height of the line in the image after applying a rotation matrix.
    Args:
        image (cv2.Mat): Input image in **grayscale**.
        rotation_matrix (cv2.Mat): Rotation matrix to apply to the image.
        black_base_line (int): The baseline height to subtract from the line height.
    Returns:
        int: Height of the line in the rotated image.
    """
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    (h, w) = image.shape
    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=255)
    cropped_height = fit_image(rotated_image, black_base_line=black_base_line)
    return cropped_height


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', )))
from BottomRowUnifier import bottom_row_unifierGRAY, bottom_row_unifier
def process_image(filepath: str, rotation_matrix: np.ndarray, cropped_height, output_path: str = None,
                  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) -> None:
    """
    Processes an image by applying a rotation matrix and saving the result.
    Args:
        file (str): Path to the input image file.
        rotation_matrix (cv2.Mat): Rotation matrix to apply to the image.
    Returns:
        None: The function saves the processed image to the same path.

    Calling image[cropped_height+10:, :] = 0  before image rotation make weird artifacts
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/BaseLine/doc/rotationweirdartifacts.png" alt="Italian Trulli">
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    (w, h) = image.shape[:2]
    if output_path is None:
        output_path = os.path.dirname(filepath).replace("frames", "frames_rotated")
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image   = cv2.warpAffine(image, rotation_matrix, (h, w ),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    rotated_image[cropped_height+10:, :] = 0  # Set the top part of the image to black

    # TODO: normalize bottom row
    _rotated_image = bottom_row_unifierGRAY(rotated_image, target_height=130)
    
     ## Close operation fills small dark holes # Kernel size depends on spot size
    # _rotated_image = cv2.morphologyEx(_rotated_image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_path, os.path.basename(filepath)), _rotated_image)

def folderBaseLineNormalizer(experiment, output_path= None):
        files = sorted(glob.glob(os.path.join(experiment, "*.png")))
        
        if output_path is None:
            output_path = os.path.dirname(files[0]).replace("frames", "frames_rotated")

            if len(glob.glob(os.path.join(output_path, "*.png"))) == len(files):
                # print(f"All images in {output_path} are already processed.")
                # return 1
                pass
            else:
                if not os.path.isdir(output_path):
                    os.makedirs(output_path, exist_ok=True)
        
        image = cv2.imread(os.path.join(experiment, files[2]), cv2.IMREAD_GRAYSCALE)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        angle,_shape, rotated_image = fit_and_rotate_image(os.path.join(experiment, files[2]),results=True)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count()*0.75)) as pool: #
            cropped_height_list = pool.starmap(line_finder, [(file, rotation_matrix) for file in files])
        cropped_height = np.array(cropped_height_list).mean().astype(np.int16)
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, cropped_height+10), angle, 1.0)
        # for file in files:
        #     process_image(file, rotation_matrix, cropped_height)

        # Rotate and save all images in parallel
        # with multiprocessing.Pool(processes=int(multiprocessing.cpu_count()*0.75)) as pool:
        #     for _ in tqdm.tqdm(pool.starmap(process_image, [(file, rotation_matrix) for file in files]), total=len(files)):
        #         pass
        with multiprocessing.Pool(processes=int(multiprocessing.cpu_count() * 0.75)) as pool:
            pool.starmap(process_image, [(file, rotation_matrix,cropped_height) for file in files])
        

if __name__ == "__main__":
    """
    I assumed Images are gray
    """
    import shutil
    for experiment in tqdm.tqdm(sorted(glob.glob("/media/d2u25/Dont/frames/*/*/*"))):
        try:
            video_experiment = experiment.replace("frames", "VideoProcess")
            outputpath = os.path.join(video_experiment, "result.mp4")
            if os.path.isfile(outputpath):
                # print(f"Video already exists for {experiment}. Skipping...")
                continue
            folderBaseLineNormalizer(experiment)

            if not os.path.isdir(video_experiment):
                os.makedirs(video_experiment, exist_ok=True)
            create_video_from_images(experiment.replace("frames", "frames_rotated"),outputpath, extension="png")

            shutil.rmtree(experiment.replace("frames", "frames_rotated"))
        except Exception as e:
            print(f"Error processing {experiment}: {e}")
            continue
        # break

    # angle, imageDim, rotated_image = fit_and_rotate_image("/media/d2u25/Dont/frames/280/S2-SNr2.1_D/T528_01/frame_000001.png", "T738_01", results=True)
    # print(f"Rotation angle: {angle} degrees")


    # '/media/d2u25/Dont/frames_rotated_rotated/285/S3-SNr3.06_D/T547_12'
    #  /media/d2u25/Dont/frames_rotated_rotated/280/S2-SNr2.1_D/T528_15