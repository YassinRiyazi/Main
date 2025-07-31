import os
import shutil
import cv2
import numpy as np
import glob
import pandas as pd
from multiprocessing import Pool, cpu_count

from PossitionalImageGenerator import sinusoidal_positional_encoding

def PE_Generator(numberOfImages:int, 
                 save_address:os.PathLike,
                  PE_height:int = 530,
                  velocity_encoding:bool = False,
                  PositionalEncoding:bool = True,
                  default_image_size:tuple = (1245, 130))-> cv2.Mat:
    """
    TODO:
        Check position encoding yields a better results or velocity encoding.
        Position encoding is to fix the width of the PE to 1246 and Velocity encoding it to calculate the length of the images and then resize width to 1245.


    Generate positional encodings for a set of images.
    args:
        numberOfImages (int): Number of images to generate positional encodings for. Basically the width of the image.
        save_address (os.PathLike): Path to save the positional encoding image.
        PE_height (int): Height of the positional encoding image. I used 530 to move embedding a little up to avoid losing it after placing the drop. Later PE resized to (130, 1248) to match the image size.
        velocity_encoding (bool): If True, use velocity encoding. Default is False.
        PositionalEncoding (bool): If True, use positional encoding. Default is True.
        default_image_size (tuple): Default size of the image to resize the positional encoding to. Default is (1245, 130).
    returns:
        pe_norm (cv2.Mat): Normalized positional encoding image.
    raises:
        ValueError: If the number of images is less than 1.
        ValueError: If both velocity_encoding and PositionalEncoding are True.
    """
    if velocity_encoding!= PositionalEncoding:
        pass
    elif velocity_encoding==PositionalEncoding:
        raise ValueError("Either velocity_encoding or PositionalEncoding must be True, not both.")

    if velocity_encoding:
        if numberOfImages < 1:
            raise ValueError("Number of images must be at least 1.")

        pe = sinusoidal_positional_encoding(numberOfImages, PE_height).T
        pe      = cv2.resize(pe, default_image_size, interpolation=cv2.INTER_LINEAR)
    elif PositionalEncoding:
        pe = sinusoidal_positional_encoding(default_image_size[0], PE_height).T
        pe      = cv2.resize(pe, default_image_size, interpolation=cv2.INTER_LINEAR)


    pe_norm = cv2.normalize(pe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imwrite(os.path.join(save_address, 'pe', 'PositionalFullGray.png'), pe_norm)
    return pe_norm

def make_PE_image(source_img:cv2.Mat,
                                 fill_img:cv2.Mat,
                                 threshold_activation:int = 1)-> cv2.Mat:
    """
    Generates a fill image based on the source image by replacing pixels in the fill image with those from the source image where the source image is below a certain threshold.
    This function modifies the fill image in place.
    args:
        source_img: Source image
        fill_img: Fill image
        threshold_activation: Threshold for activating the fill image. Default is 1.
    
    returns:        
        fill_img: The function modifies the fill image in place.

    raises:
        ValueError: If the source and fill images do not have the same dimensions.
        ValueError: If no dark regions are found in the source image.
    """


    if source_img.shape != fill_img.shape:
        fill_img = cv2.resize(fill_img, (source_img.shape[1], source_img.shape[0]))
        # raise ValueError(f"Source {source_img.shape} and fill images {fill_img.shape} must have the same dimensions. ")

    # Find all external contours
    _, binary_mask  = cv2.threshold(source_img, 20, 255, cv2.THRESH_BINARY_INV)
    contours, _     = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No dark regions found in the source image.")

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)


    # Create a mask for the largest contour
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    contour_mask = np.zeros(source_img.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    contour_mask = cv2.erode(contour_mask,np.ones((5,5),np.uint8),iterations = 3)
    

    inside = contour_mask <= threshold_activation
    fill_img[inside] = source_img[inside]
    return fill_img
    # fill_img = fill_img.reshape(*contour_mask.shape, )
    # cv2.imwrite(output_path, fill_img)

def make_PE_image_Folder(address:os.PathLike,
                         verbose:bool=False,
                         remove_Previous_Dir:bool=True)-> None:
    """
        TODO:
            - [ ] Test with super resolution images.
            - [ ] Test with padding images. Right now I'm gonna prototype with resizing the result image and feed it to the model.
    Generate positional encoding images for a folder of images.
    args:
        address: Path to the folder containing the images.
    verbose: If True, print the progress of the function.
    returns:
        None
    raises:
        ValueError: If the address does not contain any images.
        ValueError: If the index in the CSV file is NaN for any image.

    """

    images = glob.glob(os.path.join(address, '*.jpg'))
    if not images:
        raise ValueError(f"No images found in the address {address}. Please check the folder.")
    images.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

    if verbose:
        print(f"Found {len(images)} images in {address}")

    base_save_address =  address#os.path.split(address)


    save_address = base_save_address[0].replace('frame_Extracted', 'frame_position')
    if remove_Previous_Dir and os.path.exists(save_address):
        shutil.rmtree(save_address, ignore_errors=True)
    os.makedirs(os.path.join(save_address, 'pe'), exist_ok=True)

    pe_norm = PE_Generator(len(images), save_address)

    # Read the CSV file with detections
    address_df  = os.path.join(address.replace("frame_Extracted","frame_Extracted_xx"), 'detections.csv')
    df          = pd.read_csv(address_df)
    if df.empty:
        raise ValueError(f"No detections found in {address_df}. Please check the CSV file.")

    for _idx, image in enumerate(images):
        image_name = os.path.basename(image)
        IndexDF, endpoint,beginning = df.iloc[_idx]
        if IndexDF != IndexDF:
            raise ValueError(f"IndexDF is NaN for image {image_name}. Please check the CSV file.")
        
        pe_norm_cropped = pe_norm[:, endpoint:beginning]

        source_img=cv2.imread(image)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

        PE_image = make_PE_image(source_img=source_img, fill_img=pe_norm_cropped)
        if verbose:
            print(f"Processed image {image_name} with PE cropping from {endpoint} to {beginning}")

        cv2.imwrite(os.path.join(save_address, image_name), PE_image[:129,:])
        # break


def init_address(address):
    try:
        make_PE_image_Folder(address, verbose=False)
    except Exception as e:
        print(f"Failed to process {address}: {e}")

if __name__ == "__main__":
    # addresses = glob.glob('/media/d2u25/Dont/S4S-ROF/frame_Extracted/*/*/*')
    # addresses = [address for address in addresses if int(address[-2:])< 5]

    addresses = []
    for tilt in glob.glob('/media/d2u25/Dont/S4S-ROF/frame_Extracted/*'):
        for exp in glob.glob(os.path.join(tilt, '*')):
            for _ind, rep in enumerate(glob.glob(os.path.join(exp, '*'))):
                addresses.append(rep)
                if _ind > 3:
                    break

    num_workers = min(cpu_count(), 8)  # Use all CPUs or limit to 8 if too many
    with Pool(processes=num_workers) as pool:
        pool.map(init_address, addresses)



    # address = '/media/d2u25/Dont/S4S-ROF/frame_Extracted/280/S2-SNr2.1_D/frame_Extracted20250621_203528_DropNumber_01'
    # make_PE_image_Folder(address)  

    
    