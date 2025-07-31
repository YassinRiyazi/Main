import cv2
import numpy as np
from PIL import Image

def errorImage(source_img, fill_img):
    if source_img is None or fill_img is None:
        print("Error: Could not open source or fill image.")
    
    # Resize fill image to match source image if needed
    if source_img.shape[:2] != fill_img.shape[:2]:
        fill_img = cv2.resize(fill_img, (source_img.shape[1], source_img.shape[0]))
        # raise ValueError("Source and fill images must have the same dimensions.")
    return fill_img
def make_zero_pixels_transparent(source_img_path, fill_img_path, output_path):
    # Open the source and fill images
    source_img  = cv2.imread(source_img_path, cv2.IMREAD_UNCHANGED)
    fill_img    = cv2.imread(fill_img_path, cv2.IMREAD_UNCHANGED)[:,1000:]

    fill_img = errorImage(source_img, fill_img)

    # Find all external contours
    _, binary_mask  = cv2.threshold(source_img, 20, 255, cv2.THRESH_BINARY_INV)
    contours, _     = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No dark regions found.")
        return

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)


    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    contour_mask = np.zeros(source_img.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    contour_mask = cv2.erode(contour_mask,np.ones((5,5),np.uint8),iterations = 3)
    

    inside = contour_mask <= 1
    fill_img[inside] = source_img[inside]
    fill_img = fill_img.reshape(*contour_mask.shape, )
    cv2.imwrite(output_path, fill_img)

if __name__ == "__main__":
    make_zero_pixels_transparent('Projects/Viscosity/Markov/output_super_res.png',
                                'Projects/Viscosity/Markov/PositionalFullGray.png',
                                'Projects/Viscosity/Markov/output2.png')
    
