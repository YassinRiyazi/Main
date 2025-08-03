import cv2
import numpy as np

def padder(image,_disp:bool=False,
           target_height = 300, target_width = 1280):
    
    # Get the current dimensions of the image
    if      len(image.shape)==3:
        height, width, _ = image.shape
    elif    len(image.shape)==2:
        height, width = image.shape
    else:
        raise "Unknown image structure"
    # Calculate how many pixels to add
    top_padding     = (target_height - height)
    bottom_padding  = 0

    left_padding    = (target_width - width) // 2
    right_padding   = target_width - width - left_padding

    # Add padding to the image
    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Save the padded image
    # cv2.imwrite('padded_image.jpg', padded_image)

    if _disp:
        # Display the padded image (optional)
        cv2.imshow('Padded Image', padded_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return padded_image

if __name__ == "__main__":
    # Load the image using OpenCV
    image = cv2.imread('Bubble/320/S1_30per_T1_C001H001S0001/000697.jpg')
    padder(image,_disp=False)
