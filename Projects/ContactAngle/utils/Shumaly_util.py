import os
import cv2
import shutil
import natsort

# import  baseline_detection
# import  criteria_definition
import numpy as np

from .angle_detection       import *
from .criteria_definition   import *
from .edge_superres         import *
from .baseline_detection    import *

from ultralytics import YOLO

def find_reds(pic):
    # red_xs=np.where(pic[:,:,0]!=pic[:,:,1])[1]####
    # red_ys=np.where(pic[:,:,0]!=pic[:,:,1])[0]####
    red_xs=np.where(pic[:,:,0]!=pic[:,:,2])[1]
    red_ys=np.where(pic[:,:,0]!=pic[:,:,2])[0]
    return(red_xs,red_ys)

def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def load_files(ad):
    valid_extensions = {"tiff", "tif", "png", "jpg", "jpeg", "bmp", "gif", "webp"}  # Common image formats
    FileNames = []
    for file in sorted(os.listdir(ad)):
        try:
            if file.split(".")[-1].lower() in valid_extensions:
                FileNames.append(file)
        except IndexError:
            pass
    return natsort.natsorted(FileNames)

def slope_measurement(ad):
    pic_slope1=cv2.imread(os.path.join(ad,"slope","1.bmp"))
    pic_slope2=cv2.imread(os.path.join(ad,"slope","2.bmp"))
    red1_xs,red1_ys=find_reds(pic_slope1)
    red2_xs,red2_ys=find_reds(pic_slope2)
    dx=red2_xs-red1_xs
    dy=red2_ys-red1_ys
    gradian=np.arctan((dy)/(dx))
    angle=gradian*180/np.pi
    rotated1=rotate_image(pic_slope1, angle[0])
    return(angle[0],rotated1, red1_xs[0], red1_ys[0], red2_xs[0], red2_ys[0])

def make_folders(ad):
    NewFolder2=os.path.join(ad,"SR_edge")
    try:
        os.makedirs(NewFolder2)
    except:
        shutil.rmtree(NewFolder2)
        os.makedirs(NewFolder2)
    NewFolder3=os.path.join(ad,"SR_result")
    try:
        os.makedirs(NewFolder3)
    except:
        shutil.rmtree(NewFolder3)
        os.makedirs(NewFolder3)

def read_four_integers(file_path):
    with open(file_path, 'r') as file:
        line = file.readline()
        numbers = list(map(int, line.strip().split()))
        if len(numbers) != 2:
            raise ValueError("The file does not contain exactly four integers.")
        return numbers


def base_function_process(ad,name_files,file_number, model, kernel, num_px_ratio, left_polynomial_degree=3, right_polynomial_degree=2):

    # img_drop=cv2.imread(os.path.join(ad,name_files[file_number]))

    # img_drop_rotated=rotate_image(img_drop, angle)
    # # drop diff
    # diff_img=cv2.absdiff(img_drop_rotated, img_frame_rotated)
    # # drop cropping
    # BaseL=Baseline(baseline, middle_drop_height=25,drop_start_height=3*3, object_detection_threshold=40)
    # drop_reflection,*dim=BaseL.drop_cropping(diff_img, x_left_margin=30, x_right_margin=120,y_up_margin=10)
    # just_drop = diff_img[dim[2]:baseline,dim[0]:dim[1],:]

    ###Yassin
    # _img_drop   = img_drop[-200:,:,:]
    # results     = yolo_m(_img_drop, verbose=False)
    # x1,y1,x2,y2 = np.array(results[0].boxes.xyxy[:, :].cpu().numpy(), dtype=np.int32)[0]
    # just_drop   = _img_drop[y1-10:y2-7,x1-10:x2+10,:]
    # just_drop   = cv2.bitwise_not(just_drop)
    ###Yassin V2
    just_drop=cv2.imread(os.path.join(ad,name_files[file_number]))
    just_drop = just_drop[:-15,:,:]
    x1 = read_four_integers(os.path.join(ad,name_files[file_number]).replace("jpg","txt"))[0]



    #super resolution    
    upscaled_image=upscale_image(model, cv2.cvtColor(just_drop.astype('uint8'), cv2.COLOR_BGR2RGB))
    # upscaled_image=upscale_image(model, just_drop.astype('uint8'))

    #utilizing morphological transformation to remove noises
    upscaled_image=cv2.morphologyEx(np.array(upscaled_image), cv2.MORPH_CLOSE, kernel)

    #keeping just external pixels as droplet curvature
    i_list, j_list =edge_extraction( upscaled_image, thr=20)

    #extracting the desired number of pixels as input of the polynomial fitting 
    left_number_of_pixels   = int(150*num_px_ratio)
    right_number_of_pixels  = int(65*num_px_ratio)
    i_left, j_left          = advancing_pixel_selection(i_list,j_list, left_number_of_pixels=left_number_of_pixels)
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
    i_left, j_left      = advancing_pixel_selection(i_list,j_list, left_number_of_pixels=left_number_of_pixels)
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