from .preprocessing import *
from .imageloader import *
from .superResolution import *
from .edgeDetection import *

import os
from typing import TextIO, Union

def logFile(ad: Union[str, os.PathLike]): # -> TextIO
    log_path = os.path.join(ad, "error_log.txt")
    if os.path.isfile(log_path):
        os.remove(log_path)
    log_file = open(log_path, "a")
    log_file.write("#Initiate\n")
    # return log_file
