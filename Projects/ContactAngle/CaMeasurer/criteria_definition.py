import warnings
import numpy as np

#advancing angle calculations
def left_angle( i_poly_left, j_poly_left, tan_pixel_number=1):
    
    dx=i_poly_left[tan_pixel_number]-i_poly_left[0]
    dy=j_poly_left[tan_pixel_number]-j_poly_left[0]
    gradian=np.arctan((dy)/(dx))
    horizontal_angle=gradian*180/np.pi
    left_angle=90-horizontal_angle
    left_pixel_position=j_poly_left[0]    
    return(left_angle,left_pixel_position)

#receding angle calculations
def right_angle( i_poly_right, j_poly_right, tan_pixel_number=1):
    
    dx=i_poly_right[tan_pixel_number]-i_poly_right[0]
    dy=j_poly_right[tan_pixel_number]-j_poly_right[0]
    gradian=np.arctan((dy)/(dx))
    horizontal_angle=gradian*180/np.pi
    right_angle=90+horizontal_angle
    right_pixel_position=j_poly_right[0]
    return(right_angle,right_pixel_position)

#middel line angle calculations
def middle_angle( i_poly_right, j_poly_right):
    dx=i_poly_right[-2]-i_poly_right[-1]
    dy=j_poly_right[-2]-j_poly_right[-1]
    gradian=np.arctan((dy)/(dx))
    horizontal_angle=gradian*180/np.pi

    if dx<0:
        middle_angle=-horizontal_angle

    if dx>0:
        middle_angle=180+90-horizontal_angle

    middle_pixel_position=i_poly_right[-1]
    return(middle_angle,middle_pixel_position)


