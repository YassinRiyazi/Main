import shutil
import os
import cv2
import tqdm
import utils
import numpy as np
import  pandas              as      pd

from    scipy.signal        import savgol_filter
import multiprocessing

from ultralytics import YOLO

import CaMeasurer


def processs(ad,reDo = True):
    """
    
    Caution:
        I assumed drop is inside images
        I assuemd drop boundries are inside image
        Images are rotated and leveled
        Images color are inverted (cv2.bitwise_not())
        Images are colored

    
    """
    CaMeasurer.make_folders(ad)
    if os.path.isfile(os.path.join(ad,'SR_result','result.xlsx')) or not reDo:
        return None
    fps                         = 5000
    cm_on_pixel_ratio           = 0.0039062
    num_px_ratio                = (0.0039062)/cm_on_pixel_ratio
    error_handling_kernel_size  = (5,5)
    model                       = CaMeasurer.initiation()
    name_files                  = CaMeasurer.load_files(ad)
    kernel                      = np.ones(error_handling_kernel_size,np. uint8)


    adv_list, rec_list, contact_line_length_list, x_center_list, y_center_list, middle_angle_degree_list,processed_number_list=[],[],[],[],[],[],[]
    rec_angle_point_list, adv_angle_point_list=[],[]



    CaMeasurer.logFile(ad)

    #def folder_pro
    for file_number in tqdm.tqdm(range(1, len(name_files))):
        try:
            arggs = utils.base_function_process(ad,name_files,file_number, model = model, kernel = kernel, num_px_ratio=num_px_ratio)
            i_list, j_list, i_left, j_left, i_right, j_right, j_poly_left, i_poly_left, j_poly_right, i_poly_right, x_cropped, i_poly_left_rotated, j_poly_left_rotated, i_poly_right_rotated, j_poly_right_rotated = arggs
            distance = (x_cropped) * 3
            address=os.path.join(ad,'SR_edge',str(name_files[file_number]))
            adv, rec,rec_angle_point, adv_angle_point, contact_line_length, x_center, y_center, middle_angle_degree=utils.visualize(address, 
                                                                                                                                        distance+np.array(i_list),j_list,distance+np.array(i_left),j_left,distance+np.array(i_right),j_right,
                                                                                                                                        j_poly_left,distance+np.array(i_poly_left),j_poly_right,distance+np.array(i_poly_right),x_cropped,
                                                                                                                                        distance+np.array(i_poly_left_rotated), j_poly_left_rotated, distance+np.array(i_poly_right_rotated),
                                                                                                                                        j_poly_right_rotated, cm_on_pixel=cm_on_pixel_ratio, middle_line_switch=1)
            
            processed_number_list.append(int(name_files[file_number].split(".")[0].split("S0001")[-1]))
            adv_list.append(adv)
            rec_list.append(rec)
            adv_angle_point_list.append(adv_angle_point)
            rec_angle_point_list.append(rec_angle_point)
            contact_line_length_list.append(contact_line_length)
            x_center_list.append(x_center)
            y_center_list.append(y_center)
            middle_angle_degree_list.append(middle_angle_degree)

        except Exception as e:
            # print(e)
            # Append the error message to a log file
            with open(os.path.join(ad,"error_log.txt"), "a") as log_file:
                log_file.write(f'File name {os.path.join(ad,name_files[file_number])} with shape of :{cv2.imread(os.path.join(ad,"drops",name_files[file_number])).shape}' + "\n")
                pass

    vel=[]
    for i in range(len(x_center_list)-1):
        vel=vel+[x_center_list[i+1]-x_center_list[i]]

    vel=np.array(vel)*fps

    df=pd.DataFrame([processed_number_list, np.arange(0, 1/fps*len(vel), 1/fps), x_center_list, adv_list,rec_list,contact_line_length_list, y_center_list, middle_angle_degree_list, vel]).T
    df=df[:-1]

    df.columns=['file number', "time (s)", 'x_center (cm)', 'adv (degree)', 'rec (degree)', 'contact_line_length (cm)', 'y_center (cm)', 'middle_angle_degree (degree)', 'velocity (cm/s)']
    filter_size=9

    df["adv (degree)"]=savgol_filter(df["adv (degree)"], filter_size, 2)
    df["rec (degree)"]=savgol_filter(df["rec (degree)"], filter_size, 2)
    df["contact_line_length (cm)"]=savgol_filter(df["contact_line_length (cm)"], filter_size, 2)
    df["y_center (cm)"]=savgol_filter(df["y_center (cm)"], filter_size, 2)
    df["middle_angle_degree (degree)"]=savgol_filter(df["middle_angle_degree (degree)"], filter_size, 2)
    df["velocity (cm/s)"]=savgol_filter(df["velocity (cm/s)"], filter_size, 2)
    df.to_csv(os.path.join(ad, 'SR_result', 'result.csv'), index=False)
def count_files(directory):
        return sum(len(files) for _, _, files in os.walk(directory))

if __name__ == "__main__":

    processs("Projects/ContactAngle/frame_Extracted20250621_203528_DropNumber_01")
