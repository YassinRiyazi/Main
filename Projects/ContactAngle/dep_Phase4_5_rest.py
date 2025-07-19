import utils
import shutil
import tqdm
import os

if __name__ == "__main__":
    for tilt in utils.get_subdirectories(r"Processed_bubble"):
        for experiment in tqdm.tqdm(utils.get_subdirectories(tilt)):
            os.remove(os.path.join(experiment,"000000.jpg"))
            shutil.rmtree(os.path.join(experiment,"slope"))
            shutil.copytree("slope",os.path.join(experiment,"slope"))
            shutil.copy("000000.jpg",os.path.join(experiment,"000000.jpg"))