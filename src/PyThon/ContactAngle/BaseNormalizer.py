import  os
import  cv2
import  shutil
import  tqdm
import  subprocess
import  multiprocessing
 
import  numpy               as      np
import  matplotlib.pyplot   as      plt

from    skimage.measure     import  ransac, LineModelND

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

def fit_and_rotate_image(image_path: os.PathLike,
                         experiment: str = None,
                         results: bool = False) -> tuple[float, tuple, np.ndarray]:
    """
    Fits a line to the edges of the image, computes the rotation angle,
    and rotates the image to level the surface.
    Optionally saves the results as images.

    args:
        image_path (os.PathLike): Path to the input image.
        experiment (str): Name of the experiment for saving results.
        results (bool): If True, displays and saves the fitted line and rotated image.

    returns:
        angle (float): The computed rotation angle in degrees.
        image_shape (tuple): The shape of the original image.
        rotated_image (np.ndarray): The rotated image with leveled surface.

    Caution:
        - Ensure the input image is in grayscale format.
        - Check the bottom rows of the image for white pixels to avoid cropping issues.
    
    TODO:
        - Run a bottom row check to ensure the image is suitable for processing.
        
    Author:
        - Yassin Riyazi
    """

    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Detect edges using Canny
    edges = cv2.Canny(image, 50, 150)
    
    # Get edge coordinates
    y_indices, x_indices = np.where(edges > 0)
    points = np.column_stack((x_indices, y_indices))
    
    # Apply RANSAC to fit a line
    model, inliers = ransac(points, LineModelND, min_samples=2, residual_threshold=2, max_trials=1000)
    
    # Get line parameters
    line_x = np.array([min(x_indices), max(x_indices)])
    line_y = model.predict_y(line_x)
    
    # Compute angle of rotation
    dx = line_x[1] - line_x[0]
    dy = line_y[1] - line_y[0]
    angle = np.degrees(np.arctan2(dy, dx))
    # Rotate image to level the line
    (h, w) = image.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    
    if results:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.plot(line_x, line_y, color='red', linewidth=2, label='Fitted line')
        plt.legend()
        plt.title(fr"{experiment} with Fitted Line")
        
        plt.subplot(1, 2, 2)
        plt.imshow(rotated_image, cmap='gray')
        plt.title("Rotated and Cropped Image (Leveled Surface)")
        _folder = os.path.join(os.path.split(image_path)[0],"rotation")
        if not os.path.exists(_folder):
            os.mkdir(_folder)
        plt.savefig(os.path.join(_folder,"result.png"),dpi=300)
        plt.close()
    
    return angle, image.shape, rotated_image

def fit_image(image, black_base_line=10):
    
    # Detect edges using Canny
    edges = cv2.Canny(image, 50, 150)
    
    # Get edge coordinates
    y_indices, x_indices = np.where(edges > 0)
    points = np.column_stack((x_indices, y_indices))
    
    # Apply RANSAC to fit a line
    model, inliers = ransac(points, LineModelND, min_samples=2, residual_threshold=2, max_trials=1000)
    
    # Get line parameters
    line_x = np.array([min(x_indices), max(x_indices)])
    line_y = model.predict_y(line_x)
    
    # Compute angle of rotation
    dx = line_x[1] - line_x[0]
    dy = line_y[1] - line_y[0]
    angle = np.degrees(np.arctan2(dy, dx))
    return  int((line_y[1]+line_y[0])//2) - black_base_line

def line_finder(file, rotation_matrix):
    image = cv2.imread(os.path.join(experiment, file), cv2.IMREAD_GRAYSCALE)
    (h, w) = image.shape
    center = (w // 2, h // 2)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    cropped_height = fit_image(rotated_image)
    return cropped_height

def process_image(file):
    # lower_rows = 20
    image = cv2.imread(os.path.join(experiment, file), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return  # Skip if the image couldn't be loaded
    output_path = os.path.join(experiment, file)
    

    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image   = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    # cropped_height  = fit_image(rotated_image)
    # print(cropped_height)
    # rotated_image[cropped_height+3:-1, :] = 0
    # cropped_height += lower_rows

    # cropped_height = cropped_height+2
    # _img = rotated_image[:cropped_height, 30:-5]
    # print(rotated_image.shape,h, w)

    roww = 1100
    _B = np.ones(shape=(roww, 1280-35))*255
    # _B[-10:,:] = 0
    ind = 10
    _B[-ind:,:] = rotated_image[cropped_height:cropped_height+ind,30:-5]
    _B[-cropped_height-10:-10,:] = rotated_image[0:cropped_height,30:-5]
    # _img = cv2.resize(_img, (1245,1004), interpolation = cv2.INTER_LINEAR)
    cv2.imwrite(output_path, _B)

if __name__ == "__main__":
    angle, imageDim, rotated_image = fit_and_rotate_image("/media/d2u25/Dont/S4S-ROF/Teflon/280/S2-SNr2.9_D/T738_01/frame_000001.png", "T738_01", results=True)
    print(f"Rotation angle: {angle} degrees")
    # REDO = False
    # for tilt in get_subdirectories(r"Bubble"):
    #     for experiment in tqdm.tqdm(get_subdirectories(tilt)):
    #         # Run the Bash command using subprocess
    #         if not os.path.isfile(os.path.join(experiment,"000001.jpg")) or REDO:
    #             # subprocess.run(["bash", "-c", f'find "{experiment}" -type f -name "*.jpg" -delete'], check=True)
    #             # print(f"All .jpg files in subdirectories of {experiment} have been removed.")
    #             img_mkr(experiment,use_select_filter=0)

    #         # #for finding the rotation angle
    #         # files = load_files(experiment)

    #         # image = cv2.imread(os.path.join(experiment, files[10]), cv2.IMREAD_GRAYSCALE)
    #         # (h, w) = image.shape
    #         # center = (w // 2, h // 2)
    #         # angle,_shape, rotated_image = fit_and_rotate_image(os.path.join(experiment,files[10]),results=True)
    #         # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    #         # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #         #     cropped_height_list = pool.starmap(line_finder, [(file, rotation_matrix) for file in files])
    #         # cropped_height = np.array(cropped_height_list).mean().astype(np.int16)
                      
    #         # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #         #     for _ in tqdm.tqdm(pool.imap_unordered(process_image, files), total=len(files)):
    #         #         pass
            
    #         # image = cv2.imread(os.path.join(experiment, files[10]), cv2.IMREAD_GRAYSCALE)
    #         # cropped_height  = fit_image(image)
    #         # print(cropped_height)

    #         # for file in files:

    #         #     img = cv2.imread(os.path.join(experiment,file))
                
    #         #     if np.sum(img[-30,:]>=200)== img.shape[1]:
    #         #         os.remove(os.path.join(experiment,file))
    #         #         # print("drop is not in the canvas")

    #         #     elif (img[-30,1240,:]>np.array([150,150,150])).sum()==3:
    #         #         # plt.imshow(image, "grey")
    #         #         # plt.title(f"{experiment}")
    #         #         # plt.show()
    #         #         break

    #         #     else:
    #         #         os.remove(os.path.join(experiment,file))

    #         # for file in reversed(files):
    #         #     image = cv2.imread(os.path.join(experiment,file), cv2.IMREAD_GRAYSCALE)

    #         #     if np.sum(image[-70,:]>=200)== image.shape[1]:
    #         #         os.remove(os.path.join(experiment,file))
    #         #         # print("drop is not in the canvas")

    #         #     elif np.sum(image[-50,:150]>=200)>30 or np.sum(image[-50,:40]<=100)>10:
    #         #         os.remove(os.path.join(experiment,file))
    #         #         # print("partially inside canvas")
    #         #     else:
    #         #         # plt.imshow(image, "grey")
    #         #         # plt.title(f"{experiment}")
    #         #         # plt.show()
    #         #         break

    #         # shutil.copy("000000.jpg",f"{experiment}/000000.jpg")
    #         # break
    #     break


