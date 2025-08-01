"""
    Author: Yassin Riyazi
    Date: 01-07-2025
    Description: This script unifies the bottom rows of an image to a specified target height.


    [V] 1. Opening an image
    [ ] 2. counting resizing - Not necessary
    [ ] 3. finding sum of row - Not necessary
    [ ] 4. starting from bottom to top to find the number of black rows
    [V] 5. padding the image if needed
    [V] 6. returning the image
"""

import cv2
import numpy as np

def bottom_row_unifier(image, 
                       target_height=100) -> cv2.Mat:
    """
    Unifies the bottom rows of an image to a specified target height.
    args:
        image (cv2.Mat): Input image to process.
        target_height (int): Desired height of the output image. Default is 100 pixels.

    Returns:
        Processed image with unified bottom rows.

    caution:
        Resizing is mistake. 
        Do the summation in loop and stop when sum is more than one
    """
    ## Step 1: Resize the image if necessary
    image           = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image   = image[:,::50]
    vv              = resized_image.sum(axis=1)
    height = len(vv)
    """
        Resizing is mistake. 
        Do the summation in loop and stop when sum is more than one
    """
    for i in range(height-1, 0, -1):
        if vv[i] > 0:
            i -= 1
            break
    padding_top = 156 - height

    image = cv2.copyMakeBorder(image[:i,:], padding_top, 0, 0, 0, cv2.BORDER_CONSTANT, None, value = 255)
    image = cv2.copyMakeBorder(image[:,:], 0, 10, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)

    return image


if __name__ == "__main__":
    # Example usage
    image_path = "Projects/Viscosity/BottomRowUnifier/Images/000002.jpg"
    image           = cv2.imread(image_path)
    processed_image = bottom_row_unifier(image, image.shape)
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)