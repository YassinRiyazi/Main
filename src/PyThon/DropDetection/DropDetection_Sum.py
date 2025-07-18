import sys
import os

# Add the absolute path to the ./src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'src')))

# Now you can import normally
from PyThon.Performance import average_performance,measure_performance

#######################################################################################3

"""
    Caution:
        Code will fail if there are more than one drop in the image.
        Images should have exactly 5 rows of black pixels at the bottom of the image.
        It works with tilted setup drop imaeges, on other drop shape it is untested.
        It intended for leveled images, and not tested on normal images.
        If a smudge hit a drop, result are untested. 

"""

import cv2
import numpy                as np
import matplotlib.pyplot    as plt

def walk_forward(array, steep = 0.1)->int:
    """
    Walk forward from the start index until we find a value greater than 0.

    Steep: a parameter that defines the steepness of the slope to detect the drop.
    """
    for i in range(10, len(array)-10, 1):
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

    """
    resized_image = image
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Resize the image to a smaller size for faster processing
    resized_image = cv2.resize(resized_image, (dims[1]//scaleDownFactorx, dims[0]//scaleDownFactory))
    
    ## Close operation fills small dark holes # Kernel size depends on spot size
    resized_image = cv2.morphologyEx(resized_image, cv2.MORPH_CLOSE, kernel)

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

def draw_bounds(image, start, end,scaleDownFactor, thickness=2)-> None:
    """
    Draw a rectangle on the image from start to end.
    """
    plt.imshow(image)
    plt.axis('on')  # or 'off' to hide axis
    plt.axvline(x=start*scaleDownFactor,    color='red', linewidth=thickness)
    plt.axvline(x=end*scaleDownFactor,      color='red', linewidth=thickness)

    plt.show()

def detection(image,scaleDownFactor):
    vv              = detect_drop(image,image.shape, show=False, scaleDownFactorx=scaleDownFactor)
    endpoint        = walk_forward(vv)
    beginning       = backward(vv)
    return beginning, endpoint

@average_performance(runs=1_000)
def detection_perf(image,scaleDownFactor):
    vv              = detect_drop(image,image.shape, show=False, scaleDownFactorx=scaleDownFactor)
    endpoint        = walk_forward(vv)
    beginning       = backward(vv)
    return beginning, endpoint

if __name__ == "__main__":
    # Load the image
    scaleDownFactor = 5
    image_path      = "Projects/Viscosity/DropDetection/SampleImages/002667.jpg"
    image           = cv2.imread(image_path)
    beginning, endpoint = detection_perf(image, scaleDownFactor)

    draw_bounds(image, beginning, endpoint,scaleDownFactor)

"""
        To make code versitile, normalize based on the nimber of height pixels with maximum brighness./ otherwise overflow may occur.
            I failed, I don't know how does it generate 130000 from summing pixels.
            And it is really costly to cast float to int.

        Add timing decolator:
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
                !!! And probably inside numy transpose happends for calculating the sum.
    Code execution is 480 microseconds.

    TODO:
        Going to test with C.
"""