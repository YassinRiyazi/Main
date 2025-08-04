"""
    Author: Yassin Riyazi
    Date: 04-08-2025
    Description:
        This script processes directories containing images of drops, extracting and analyzing them using the DropDetection_Sum module.
        It utilizes multiprocessing to handle multiple directories concurrently and employs tqdm for progress tracking.
        It is designed to work with a specific directory structure and image processing requirements.
        It is part of the P2NeuralNetwork project under the Viscosity project umbrella.
"""
import  glob
import  os
import  cv2
import  numpy                   as      np
import  matplotlib.pyplot       as      plt
from    multiprocessing         import  Pool
from    tqdm                    import  tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'src/PyThon/ContactAngle/DropDetection')))
from DropDetection_Sum import Main


def worker(args):
    """Worker function to call Main with the given arguments."""
    try:
        Main(
            experiment=args['experiment'],
            SaveAddress=args['SaveAddress'],
            SaveAddressCSV=args['SaveAddressCSV'],
            extension=args['extension'],
            _morphologyEx=args['_morphologyEx']
        )
    except Exception as e:
        print(f"Error processing {args['experiment']}: {e}")

if __name__ == "__main__":

    # for tilt in glob.glob("/media/d2u25/Dont/frames_Process_15/*"):
    #     for experiment in glob.glob(os.path.join(tilt,'*')):
    #         for _idx, rep in enumerate(glob.glob(os.path.join(experiment,'*'))):
    #             if _idx < 5:
    #                 _SaveAddresses = (rep.replace('frames_Process_15', 'frames_Process_15_Patch'))
    #                 Main(experiment = rep,
    #                     SaveAddress = _SaveAddresses,
    #                     SaveAddressCSV= _SaveAddresses,
    #                     extension = '.png',
    #                     _morphologyEx = True,)

    
    # Set the number of processes (adjust as needed)
    num_processes = 14  # Example: Use 4 processes

    # Collect all tasks
    tasks = []
    for tilt in glob.glob("/media/d2u25/Dont/frames_Process_15/*"):
        for experiment in glob.glob(os.path.join(tilt, '*')):
            for _idx, rep in enumerate(glob.glob(os.path.join(experiment, '*'))):
                if _idx < 5:
                    _SaveAddresses = rep.replace('frames_Process_15', 'frames_Process_15_Patch')
                    tasks.append({
                        'experiment': rep,
                        'SaveAddress': _SaveAddresses,
                        'SaveAddressCSV': _SaveAddresses,
                        'extension': '.png',
                        '_morphologyEx': True
                    })

    # Run tasks in parallel with progress bar
    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(worker, tasks), total=len(tasks), desc="Processing videos"))

    # """
    #     Check the YOLO result with OpenCV vcountors
    #     Normalize the white lines in bottom of the images
    #     save x1 in the textfile with same name as the image
    # """
