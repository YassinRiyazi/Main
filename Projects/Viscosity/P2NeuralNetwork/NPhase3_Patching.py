import  os
import  glob
import  numpy           as      np
from    tqdm            import  tqdm
from    multiprocessing import  Pool, cpu_count

import  sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'src/PyThon/Viscosity/PositionalEncoding')))
from PossitionalImageGenerator import make_PE_image_Folder

def worker(args):
    try:
        make_PE_image_Folder(address =              args['address'],
                             verbose =              args['verbose'],
                             extension =            args['extension'],
                             remove_Previous_Dir =  args['remove_Previous_Dir'],
                             velocity_encoding   =  args['velocity_encoding'],
                             positional_encoding =  args['positional_encoding'])
    except Exception as e:
        print(f"Failed to process {args['address']}: {e}")

if __name__ == "__main__":
    tasks = []
    for tilt in sorted(glob.glob('/media/d2u25/Dont/frames_Process_15_Patch/*')):
        for exp in sorted(glob.glob(os.path.join(tilt, '*'))):
            for _ind, rep in enumerate(sorted(glob.glob(os.path.join(exp, '*')))):
                tasks.append({
                        'address': rep,
                        'verbose': False,
                        'extension': '.png',
                        'remove_Previous_Dir': False,
                        'velocity_encoding': False,
                        'positional_encoding': True
                    })

    num_processes = 14  # Example: Use 14 processes
    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(worker, tasks), total=len(tasks), desc="Processing videos"))