import tqdm
import utils
import multiprocessing
import glob
def process_experiment(experiment):
    utils.img_mkr(experiment, use_select_filter=0)

if __name__ == "__main__":
    experiments = []
    for tilt in utils.get_subdirectories(r"drop"):
        for fluid in utils.get_subdirectories(tilt):
            for video in sorted(glob.glob(f"{fluid}/*.mp4")):
                experiments.append(video)


    # Use multiprocessing pool
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_experiment, experiments), total=len(experiments)))


"""
Change log

V2.0.0
    Added cuda acceleration
    Added Multiprocess

    GPU Utils:60%
    CPU Utils:99%
    8H-> 10M
V1.0.0
    Initiated
    GPU Utils:00%
    CPU Utils:30%
"""