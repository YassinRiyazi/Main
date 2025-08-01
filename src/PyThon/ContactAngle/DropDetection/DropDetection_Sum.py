"""
    Author: Yassin Riyazi
    Date: 01-07-2025
    Description: This script detects drops in images by analyzing pixel intensity changes.

    Caution:
        Code will fail if there are more than one drop in the image.
        Images should have exactly 5 rows of black pixels at the bottom of the image.
        It works with tilted setup drop images, on other drop shape it is untested.
        It intended for leveled images, and not tested on normal images.
        If a smudge hit a drop, result are untested. 


    To make code versatile, normalize based on the number of height pixels with maximum brightness./ otherwise overflow may occur.
            I failed, I don't know how does it generate 130000 from summing pixels.
            And it is really costly to cast float to int.

        Add timing decorator:
            - Test general time
                [detection] Ran 10000 times
                [detection] Avg execution time: 0.000407 seconds
                [detection] Avg peak memory usage: 158.18 KiB

            - Optimize image resizing and processing time.

            - Check effect of transposing on performance.
                    [detection] Ran 10000 times
                    [detection] Avg execution time: 0.000407 seconds
                    [detection] Avg peak memory usage: 158.18 KiB

                With if with Transpose:
                    [detection] Ran 10000 times
                    [detection] Avg execution time: 0.000477 seconds
                    [detection] Avg peak memory usage: 158.18 KiB

                with if without Transpose:
                    [detection] Ran 10000 times
                    [detection] Avg execution time: 0.000475 seconds
                    [detection] Avg peak memory usage: 158.18 KiB
                !!! If is more damaging than transpose, because it is not optimized for numpy arrays.
                !!! And probably inside numpy transpose happens for calculating the sum.
    Code execution is 480 microseconds.
"""

import cv2
import numpy                as np
import matplotlib.pyplot    as plt

def walk_forward(array: np.ndarray,
                 steep: float = 0.1) -> int:
    """
    Walk forward from the start index until we find a value greater than 0.

    Steep: a parameter that defines the steepness of the slope to detect the drop.

    args:
        array (np.ndarray): The array to walk through.
        steep (float): The steepness of the slope to detect the drop.
    returns:
        int: The index where the drop is detected.
    """
    for i in range(10, len(array)-10, 1):
        if np.abs(array[i] - array[i-1])> steep:
            return i - 1
    raise ValueError("No drop detected in the image. Please check the image or parameters.")
    return None

def backward(array: np.ndarray,
             steep: float = 0.0025) -> int:
    """
    Walk forward from the start index until we find a value greater than 0.

    Steep: a parameter that defines the steepness of the slope to detect the drop.

    args:
        array (np.ndarray): The array to walk through.
        steep (float): The steepness of the slope to detect the drop.
    returns:
        int: The index where the drop is detected.

    if want to include wider dropes with low slope decrese the sensitivity of the slope.
    For example, if you want to detect drops with low surface tention, surfactant, slope shold be less than 0.0025
    """
    for i in range(len(array)-10, 10, -1):
        if np.abs(array[i] - array[i-1])> steep:
            return i
        
    raise ValueError("No drop detected in the image. Please check the image or parameters.")
    return None
       
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

def detect_drop(image:cv2.Mat,
                dims:tuple[int, int],
                show:bool = False,
                scaleDownFactorx:int = 5, 
                scaleDownFactory:int = 2)-> np.ndarray:
    """
    Trying to detect images with less than 2ms delay.

    [V] 0. Convert image to grayscale,
    [V] 1. Resizing image,
    [V] 2. Applying gaussian blur (morphologyEx worked better than Gaussian blur),
    [-] 3. Transposing image (Optimization purposes) [Didn't improve any thing and because of if damaged was more than benefits],
    [V] 4. Summation over rows
    [-] 5. Normalize images based on height and maximum brightness,
    [V] 6. Finding beginning of the drop and ending of the drop,
    and finally drawing a rectangle around the drop.

    args:
        image (cv2.Mat): Input image to detect drops.
        dims (tuple[int, int]): Dimensions of the input image.
        show (bool): Whether to show the processed image and plot the sum of rows.
        scaleDownFactorx (int): Factor to scale down the width of the image.
        scaleDownFactory (int): Factor to scale down the height of the image.
    
    returns:
        np.ndarray: Sum of rows of the processed image.

    Caution:
        Code will fail if there are more than one drop in the image.
        Images should have exactly 5 rows of black pixels at the bottom of the image.
        It works with tilted setup drop images, on other drop shape it is untested

    TODO:
        Going to test with C.
    """
    resized_image = image
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Resize the image to a smaller size for faster processing
    resized_image = cv2.resize(resized_image, (dims[1]//scaleDownFactorx, dims[0]//scaleDownFactory))
    
    ## Close operation fills small dark holes # Kernel size depends on spot size
    resized_image = cv2.morphologyEx(resized_image, cv2.MORPH_CLOSE, kernel)
    """
    Opening is just another name of erosion followed by dilation. 
    It is useful in removing noise, as we explained above. Here we use the function,
    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    """

    vv = resized_image.sum(axis=0)
    # vv = vv/(dims[0]-15)/scaleDownFactory  # Normalize the sum of rows
    vv = vv/vv.mean()  # Normalize the sum of rows to have a mean of 1

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

    return vv

def draw_bounds(image:cv2.Mat,
                start:int, end:int,
                scaleDownFactor:int, thickness:int=2) -> None:
    """
    Draw a rectangle on the image from start to end. 
    For testing purposes, it draws a rectangle on the image to visualize the detected drop.
    Args:
        image (cv2.Mat): Input image to draw the rectangle on.
        start (int): Starting index of the rectangle.
        end (int): Ending index of the rectangle.
        scaleDownFactor (int): Factor to scale down the image.
        thickness (int): Thickness of the rectangle border.

    Returns:
        None: Displays the image with the rectangle drawn.
    """
    plt.imshow(image)
    plt.axis('on')  # or 'off' to hide axis
    plt.axvline(x=start*scaleDownFactor,    color='red', linewidth=thickness)
    plt.axvline(x=end*scaleDownFactor,      color='red', linewidth=thickness)

    plt.show()

def detection(image:cv2.Mat,
              scaleDownFactor:int) -> tuple[int, int]:
    """
    Detects the drop in the image by analyzing pixel intensity changes.
    For testing purposes, it returns the indices of the beginning and end of the drop.
    Args:
        image (cv2.Mat): Input image to detect drops.
        scaleDownFactor (int): Factor to scale down the image for processing.   
    Returns:
        tuple[int, int]: Indices of the beginning and end of the drop in the image.
    """
    vv              = detect_drop(image,image.shape, show=False, scaleDownFactorx=scaleDownFactor)
    endpoint        = walk_forward(vv)
    beginning       = backward(vv)
    return beginning, endpoint

if __name__ == "__main__":
    import sys
    import os

    # Add the absolute path to the ./src folder
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'src')))

    # Now you can import normally
    from PyThon.Performance import average_performance,measure_performance

    @average_performance(runs=1_000)
    def detection_perf(image,scaleDownFactor):
        vv              = detect_drop(image,image.shape, show=False, scaleDownFactorx=scaleDownFactor)
        endpoint        = walk_forward(vv)
        beginning       = backward(vv)
        return beginning, endpoint
    

    # Load the image
    scaleDownFactor = 5
    image_path      = "Projects/Viscosity/DropDetection/SampleImages/002667.jpg"
    image           = cv2.imread(image_path)
    beginning, endpoint = detection_perf(image, scaleDownFactor)
    
    draw_bounds(image, beginning, endpoint,scaleDownFactor)