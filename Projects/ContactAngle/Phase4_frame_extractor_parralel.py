import shutil
import  os
import  cv2
import  shutil
import  tqdm
import  utils
import  multiprocessing
 
import  numpy               as      np
import  matplotlib.pyplot   as      plt

from    skimage.measure     import  ransac, LineModelND

def process_file(file, experiment, new_address,target_height, target_width):
    """
    Process a single file by reading, padding, and saving the image.
    
    :param file: File name to be processed.
    :param experiment: Path to the experiment directory.
    :param new_address: Directory where processed images should be saved.
    """
    image = cv2.imread(os.path.join(experiment, file), cv2.IMREAD_GRAYSCALE)
    new_image = utils.padder(image,target_height = target_height, target_width = target_width)
    cv2.imwrite(os.path.join(new_address, file), new_image)

def process_experiment(tilt, experiment,root_folder_name,target_height, target_width):
    """
    Process all files in an experiment directory in parallel using multiprocessing.
    
    :param tilt: Path to the tilt directory.
    :param experiment: Path to the experiment directory.
    """
    new_address = os.path.join(root_folder_name, tilt.split("/")[-1], experiment.split("/")[-1])
    utils.ensure_directory_exists(new_address)

    files = utils.load_files(experiment)

    # Create a pool of workers to process the files in parallel
    with multiprocessing.Pool() as pool:
        pool.starmap(process_file, [(file, experiment, new_address,target_height, target_width) for file in files])

        pool.close()  # Close the pool to prevent any more tasks from being submitted
        pool.join()   # Wait for all worker processes to finish


def fit_and_rotate_image(image_path,experiment:str=None,
                         results:bool=False):
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
    
    # Crop the lower part of the image, keeping only the top 5 rows
    rotated_image = rotated_image[:, :]
    
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
    # center = (w // 2, h // 2)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    cropped_height = fit_image(rotated_image)
    return cropped_height

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
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
    # _img = 
    # print(rotated_image.shape,h, w)

    rotated_image = rotated_image[:cropped_height+5, 30:-5]
    rotated_image = cv2.copyMakeBorder(rotated_image, 0, 15, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # rotated_image = cv2.copyMakeBorder(rotated_image, 300 - rotated_image.shape[0], 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    rotated_image = cv2.morphologyEx(rotated_image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(output_path, rotated_image)


def clean_folder_if_processed(directory):
    # Traverse through the directory and its subdirectories
    for root, dirs, files in os.walk(directory, topdown=False):
        # Check if '.processed' file exists in the current directory
        if '.processed' in files:
            return True
        else:
            return False




if __name__ == "__main__":
    destfolder    = 'Processed'
    root_folder_name = "frames"
    target_height       = 160
    target_width        = 1280

    REDO = True
    for tilt in utils.get_subdirectories(root_folder_name):
        for fluid in utils.get_subdirectories(tilt):
            for experiment in utils.get_subdirectories(fluid):
                alt_adress   = experiment.replace(root_folder_name, destfolder)

                if os.path.isfile(os.path.join(experiment,".processed")):
                    print(f"{experiment} - skipped.")
                    continue
                else:
                    try:
                        process_experiment(tilt, alt_adress,root_folder_name,target_height, target_width)

                        #for finding the rotation angle
                        files = utils.load_files(experiment)
                        # image = cv2.imread(os.path.join(experiment, files[10]), cv2.IMREAD_GRAYSCALE)
                        # (h, w) = image.shape
                        h,w = target_height,target_width
                        center = (w // 2, h // 2)

                        angle,_shape, rotated_image = fit_and_rotate_image(os.path.join(experiment,files[10]),results=True)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

                        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                            cropped_height_list = pool.starmap(line_finder, [(file, rotation_matrix) for file in files])
                        cropped_height = np.array(cropped_height_list).mean().astype(np.int16)
                                
                        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                            for _ in tqdm.tqdm(pool.imap_unordered(process_image, files), total=len(files)):
                                pass
                            pool.close()  # Close the pool to prevent any more tasks from being submitted
                            pool.join()   # Wait for all worker processes to finish

                        utils.writter(os.path.join(experiment,".processed"))
                    except:
                        print(f"Processing {experiment} with files")




"""
Changelog:





Version 1.0.0
    I tried really hard to resolve padding yet failed.
    "roww = 1100
        _B = np.ones(shape=(roww, 1280-35))*255
        # _B[-10:,:] = 0
        ind = 10
        _B[-ind:,:] = rotated_image[cropped_height:cropped_height+ind,30:-5]
        _B[-cropped_height-10:-10,:] = rotated_image[0:cropped_height,30:-5]
        # _img = cv2.resize(_img, (1245,1004), interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(output_path, _B)"

    insted use cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    First folder 46 minutes

"""