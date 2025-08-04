import  cv2
import  numpy               as  np
import  matplotlib.pyplot   as  plt
from    PIL                 import Image
import  os
import  pandas              as  pd
import  shutil
import  glob
from    multiprocessing     import Pool, cpu_count
from    tqdm                import tqdm
def sinusoidal_positional_encoding(max_position, d_model):
    """
    Task:       
        This is a part of viscosity estimation project.
    Sub-Task:  
        Implement positional encoding for the Markov chain model.

    Description:
        Probably I am going to use CNN and Transformer for this project. I will treat each frame of the video as
        a Markov state and the video as a Markov chain.
        But I need to provide position of the object and its velocity as input features to the model.
        I want to use positional encoding to encode the position and velocity of the object in the video.
    
        Before I have used padding to keep the scale of objects in frames.
        I was thinking about by adding polar/arc like positional encoding maybe I could keep the scale of objects in frames.
        But I am not sure if it is a good idea.
        I will try to implement it and see how it works. Worst case I will use padding again.

    Why positional encoding?
        In the context of transformers, every word in a sentence is mapped to a vector through embeddings. The transformer then uses these vectors to generate keys and queries for its self-attention mechanism.
        The effectiveness of this process hinges on how well the positional encoding adapts to shifts in position., which do not inherently understand the order of input data.
        In a Markov chain model, especially when dealing with time series or sequential data, positional encoding helps the model to:
        - Understand the temporal relationships between states.
        - Capture the dynamics of the system over time.
        - Maintain the order of states in the Markov chain, which is essential for accurate predictions
        - Facilitate the model's ability to generalize from seen to unseen positions in the sequence
        
    Steps:
        1. Deciding on the positional encoding scheme.
            one of the most important concepts of a transformer model — positional encoding. 
            The authors of “Attention is all you need” chose a combination of sinusoidal curves for the positional encoding vectors.

            Encoding must be able to identified the position of a data in a timeseries uniquely. 
            It should be able to expand to arbitrary number of data in a time series.
            It should be compact and efficient. [Not one-hot encoding]
            Binary encoding is not good because it is not smooth and does not generalize well.
                Meaningful encoding: 
                    we want small changes in position to correspond to small changes in representation.
                    The binary method, however, leads to large jumps in encoding for adjacent positions, which can confuse the model as it tries to interpret these as meaningful patterns.

                Continuity:
                    Continuity in positional encoding helps the model to generalize better to positions it hasn’t seen. A continuous encoding scheme means that positions that are numerically close will receive similar representations, and this would aid the model in understanding patterns that span across multiple positions. Lack of smooth transitions also means that the model’s ability to interpolate and extract useful positional information is limited
            
            Encoding scheme should not blow up the size of the input data. So a periodic function is a good choice.
            To extend the encoding to arbitrary number of data in a time series, we can add a perpendicular sine wave with different frequency component to the encoding.
            The first sine wave provides a base positional signal, and by adding a second sine wave at a different frequency, we allow the encoding to oscillate differently as we move along the sequence.

            This additional dimension effectively creates a more complex positional landscape.
            Each position is now represented by a point in this two-dimensional space, where each dimension corresponds to a sine wave of a different frequency.

            * My initial thought was to use the row as the position but I think its better to include time or a dimensionless number of time as the position.

            For a small positional shift δx, the change in positional encoding P should be a linear function.
            Such a transformation ensures that the relationship captured between different parts of the sentence remains consistent, even as inputs vary slightly.
            This consistency is vital for the transformer to maintain syntactic relationships within a sentence across different positions.
            

            x: Length/ Embedding dimension
            y: Speed / Position
            Finding maximum and minimum values of the frame lengths and speeds.

        2. Implement positional encoding for the Markov chain model.
        3. Test the positional encoding with a simple model.
        
                
    Analogies to LNP:
        A sentence:                 A frame of video
        A Token embedding vector:   A row of pixels in a frame
    
    References:
        https://medium.com/@gunjassingh/positional-encoding-in-transformers-a-visual-and-intuitive-guide-0761e655cea7
        https://medium.com/data-science/master-positional-encoding-part-i-63c05d90a0c3
    """

    position = np.arange(max_position)[:, np.newaxis]
    # The original formula pos / 10000^(2i/d_model) is equivalent to pos * (1 / 10000^(2i/d_model)).
    # I use the below version for numerical stability
    div_term = np.exp(np.arange(0, d_model, 2) * - (np.log(100000.0) / d_model))
    
    pe = np.zeros((max_position, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

def cropped_position_encoding(  x1,x2,
                                max_position = 1245,  # Maximum sequence length
                                d_model = 530  ,      # Embedding dimension:
                            ):
    """
    Assumptions:
        Images are unified in size ex. (130, 1248)
    
    Adding position encoding to the images

    For erode results
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/Viscosity/PositionalEncoding/doc/WithErode.png" alt="Italian Trulli">

    And without erode results see the images below.
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/Viscosity/PositionalEncoding/doc/WithoutErode.png" alt="Italian Trulli">

    """
    
    pe = sinusoidal_positional_encoding(max_position, d_model).T
    pe = cv2.resize(pe[:, x1:x2], (x2-x1-1, 130), interpolation=cv2.INTER_LINEAR)
    pe_norm = cv2.normalize(pe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create a red-black RGB image
    red_img = np.zeros((pe_norm.shape[0], pe_norm.shape[1], 3), dtype=np.uint8)
    red_img[..., 0] = pe_norm  # Red channel
    red_img[..., 1] = pe_norm  # green channel

    result_img = Image.fromarray(red_img)
    result_img.save('Projects/Viscosity/Markov/fsgfg.png')
    return pe

def main_Visualizer(max_position = 1245,  # Maximum sequence length
         d_model = 530,        # Embedding dimension
         _plot:bool = False):
    """
    Main function to generate sinusoidal positional encoding.
    """

    # Generate positional encoding
    pe = sinusoidal_positional_encoding(max_position, d_model).T
    pe = cv2.normalize(pe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if _plot:
        # Visualize the positional encoding
        plt.figure(figsize=(10, 5))
        plt.imshow(pe, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Sinusoidal Positional Encoding')
        plt.xlabel('Position')
        plt.ylabel('Embedding Dimension')
        plt.show()

    return pe

def PE_Generator(numberOfImages:int, 
                 save_address:os.PathLike,
                  PE_height:int = 530,
                  velocity_encoding:bool = False,
                  positional_encoding:bool = True,
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
        positional_encoding (bool): If True, use positional encoding. Default is True.
        default_image_size (tuple): Default size of the image to resize the positional encoding to. Default is (1245, 130).
    returns:
        pe_norm (cv2.Mat): Normalized positional encoding image.
    raises:
        ValueError: If the number of images is less than 1.
        ValueError: If both velocity_encoding and positional_encoding are True.
    """
    if velocity_encoding!= positional_encoding:
        pass
    elif velocity_encoding==positional_encoding:
        raise ValueError("Either velocity_encoding or positional_encoding must be True, not both.")

    if velocity_encoding:
        if numberOfImages < 1:
            raise ValueError("Number of images must be at least 1.")
        pe = sinusoidal_positional_encoding(numberOfImages, PE_height).T
        pe      = cv2.resize(pe, default_image_size, interpolation=cv2.INTER_LINEAR)
    elif positional_encoding:
        pe = sinusoidal_positional_encoding(default_image_size[0], PE_height).T
        pe      = cv2.resize(pe, default_image_size, interpolation=cv2.INTER_LINEAR)


    pe_norm = cv2.normalize(pe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imwrite(os.path.join(save_address, 'pe', 'PositionalFullGray.png'), pe_norm)
    return pe_norm

def make_PE_image(  source_img:np.ndarray,
                    fill_img:np.ndarray,
                    threshold_activation:int = 1)-> np.ndarray:
    """
    Generates a fill image based on the source image by replacing pixels in the fill image with those from the source image where the source image is below a certain threshold.
    This function modifies the fill image in place.
    args:
        source_img (np.ndarray): Source image
        fill_img (np.ndarray): Fill image
        threshold_activation (int): Threshold for activating the fill image. Default is 1.

    returns:
        fill_img (np.ndarray): The function modifies the fill image in place.

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
                         extension:str='.png',
                         remove_Previous_Dir:bool=False,
                         velocity_encoding: bool = False,
                         positional_encoding: bool = True)-> None:
    """
    Generate positional encoding images for a folder of images.
    
        TODO:
            - [ ] Test with super resolution images.
            - [ ] Test with padding images. Right now I'm gonna prototype with resizing the result image and feed it to the model.
            - [ ] Later test with velocity encoding.

    args:
        address (os.PathLike): Path to the folder containing the images.
        verbose (bool): If True, print the progress of the function.
        extension (str): File extension of the images to process. Default is '.png'.
        remove_Previous_Dir (bool): If True, remove the previous directory if it exists
        velocity_encoding (bool): If True, use velocity encoding. Default is False.
        positional_encoding (bool): If True, use positional encoding. Default is True.

    returns:
        None: No return value, the function saves the positional encoding images to the specified address.

    raises:
        ValueError: If the address does not contain any images.
        ValueError: If the index in the CSV file is NaN for any image.

    """

    images = glob.glob(os.path.join(address, '*' + extension))

    if not images:
        raise ValueError(f"No images found in the address {address}. Please check the folder.")
    
    images.sort()  # Sort images to maintain order

    if verbose:
        print(f"Found {len(images)} images in {address}")


    save_address = address

    if remove_Previous_Dir and os.path.exists(save_address):
        shutil.rmtree(save_address, ignore_errors=True)
    if not os.path.exists(save_address):
        os.makedirs(save_address, exist_ok=True)
    

    pe_norm = PE_Generator(len(images), save_address,
                           velocity_encoding=velocity_encoding,
                           positional_encoding=positional_encoding)

    # Read the CSV file with detections
    address_df  = os.path.join(address, 'detections.csv')
    df          = pd.read_csv(address_df)
    if df.empty:
        raise ValueError(f"No detections found in {address_df}. Please check the CSV file.")

    for _idx, image in enumerate(images):
        image_name = os.path.basename(image)

        IndexDF, endpoint, beginning = df.iloc[_idx]
        if IndexDF != IndexDF:
            raise ValueError(f"IndexDF is NaN for image {image_name}. Please check the CSV file.")
        
        pe_norm_cropped = pe_norm[:, endpoint:beginning]

        source_img=cv2.imread(image)
        if source_img is None:
            raise ValueError(f"Could not read image {image_name}. Please check the image file.")
        if len(source_img.shape) == 3:
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

        PE_image = make_PE_image(source_img=source_img, fill_img=pe_norm_cropped)
        if verbose:
            print(f"Processed image {image_name} with PE cropping from {endpoint} to {beginning}")

        cv2.imwrite(os.path.join(save_address, image_name), PE_image)

if __name__ == "__main__":
    pe = main_Visualizer(max_position = 31245, d_model = 530, _plot = False)
    result_img = Image.fromarray(pe.astype(np.uint8))
    result_img.save('Projects/Viscosity/Markov/PositionalFullGray.png')
    print("Positional encoding image saved.")