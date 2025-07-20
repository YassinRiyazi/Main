import os
import cv2
import shutil
import CaMeasurer


import numpy as np

from .angle_detection       import *
from .criteria_definition   import *
from .edge_superres         import *
from .baseline_detection    import *



# USE one in CaMeasure
def read_four_integers(file_path):
    with open(file_path, 'r') as file:
        line = file.readline()
        numbers = list(map(int, line.strip().split()))
        if len(numbers) != 2:
            raise ValueError("The file does not contain exactly four integers.")
        return numbers


def base_function_process(ad,name_files,file_number, model, kernel, num_px_ratio, left_polynomial_degree=3, right_polynomial_degree=2):
    """
        1   Loading data
        1.1 Loading the image
        1.2 cropping the base line
        1.3 loading the x1, x2 positions

        2   Supper resulotion

        3   lk
    
    """
    just_drop   = cv2.imread(os.path.join(ad,name_files[file_number]))
    just_drop   = just_drop[:-15,:,:]
    x1          = read_four_integers(os.path.join(ad,name_files[file_number]).replace("jpg","txt"))[0]


    #super resolution    
    upscaled_image = CaMeasurer.upscale_image(model, just_drop, kernel)

    

    #keeping just external pixels as droplet curvature
    i_list, j_list = CaMeasurer.edge_extraction( upscaled_image, thr=20)

    #extracting the desired number of pixels as input of the polynomial fitting 
    left_number_of_pixels   = int(64*num_px_ratio)
    right_number_of_pixels  = int(65*num_px_ratio)
    i_left, j_left          = CaMeasurer.advancing_pixel_selection(i_list,j_list, left_number_of_pixels=left_number_of_pixels)
    i_right, j_right        = receding_pixel_selection(i_list,j_list, right_number_of_pixels=right_number_of_pixels)

    #rotation for fitting, it can increase the accuracy to rotate 90 degrees then fit the polynomial
    i_left_rotated,j_left_rotated=j_left,i_left       
    i_right_rotated,j_right_rotated=j_right,i_right   

    
    i_poly_left_rotated, j_poly_left_rotated    = poly_fitting(i_left_rotated,j_left_rotated,polynomial_degree=left_polynomial_degree,line_space=left_number_of_pixels)
    i_poly_right_rotated, j_poly_right_rotated  = poly_fitting(i_right_rotated,j_right_rotated,polynomial_degree=right_polynomial_degree,line_space=right_number_of_pixels)

    right_angle_degree,right_angle_point        = right_angle(i_poly_right_rotated, j_poly_right_rotated,1)
    left_angle_degree,left_angle_point          = left_angle(i_poly_left_rotated, j_poly_left_rotated,1)
    

    if left_angle_degree<=60 :
        left_number_of_pixels=int(60*num_px_ratio)
        left_polynomial_degree=2
    elif 60<left_angle_degree<=105:
        left_number_of_pixels=int(85*num_px_ratio)
        left_polynomial_degree=2
    elif 105<left_angle_degree<=135:
        left_number_of_pixels=int(125*num_px_ratio)  #175
        left_polynomial_degree=3
    elif 135<left_angle_degree:
        left_number_of_pixels=int(145*num_px_ratio) #215
        left_polynomial_degree=4

    if right_angle_degree<=60:
        right_number_of_pixels=int(60*num_px_ratio)
        right_polynomial_degree=2
    elif 60<right_angle_degree<=105:
        right_number_of_pixels=int(85*num_px_ratio)
        right_polynomial_degree=2
    elif 105<right_angle_degree<=135:
        right_number_of_pixels=int(125*num_px_ratio) #175
        right_polynomial_degree=3
    elif 135<right_angle_degree:
        right_number_of_pixels=int(145*num_px_ratio) #215
        right_polynomial_degree=4

    #9. extracting the desired number of pixels as input of the polynomial fitting 
    i_left, j_left      = CaMeasurer.advancing_pixel_selection(i_list,j_list, left_number_of_pixels=left_number_of_pixels)
    i_right, j_right    = receding_pixel_selection(i_list,j_list, right_number_of_pixels=right_number_of_pixels)

    #10. rotation for fitting, it can increase the accuracy to rotate 90 degrees and then fit the polynomial
    i_left_rotated,j_left_rotated=j_left,i_left       
    i_right_rotated,j_right_rotated=j_right,i_right   
    i_poly_left_rotated, j_poly_left_rotated    = poly_fitting(i_left_rotated,j_left_rotated,polynomial_degree=left_polynomial_degree,line_space=left_number_of_pixels)
    i_poly_right_rotated, j_poly_right_rotated  = poly_fitting(i_right_rotated,j_right_rotated,polynomial_degree=right_polynomial_degree,line_space=right_number_of_pixels)
    j_poly_left=i_poly_left_rotated
    i_poly_left=j_poly_left_rotated
    j_poly_right=i_poly_right_rotated
    i_poly_right=j_poly_right_rotated
    # x_cropped=dim[0]
    x_cropped = x1
    return i_list, j_list, i_left, j_left, i_right, j_right, j_poly_left, i_poly_left, j_poly_right, i_poly_right, x_cropped, i_poly_left_rotated, j_poly_left_rotated, i_poly_right_rotated, j_poly_right_rotated