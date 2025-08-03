"""
    Author: Yassin Riyazi
    Date: 01-07-2025
    Description: This script exports YOLO models to TensorRT format. 
    YOLO models are trained by Yassin Riyazi for contact angle detection.
    Usage: Run this script to export the models.

"""
import os
from ultralytics import YOLO

# Load the YOLO11 model
BaseAddress = os.path.abspath(os.path.dirname(__file__))
model = YOLO(os.path.join(BaseAddress, "Gray-320-s.pt"))

# Export the model to TensorRT format
model.export(format="engine",
            imgsz=(640, 640),
            dynamic=False,
            #  int8=True,
            batch=1,
            #  half=True,
             verbose=False,
             simplify=True)

# Load the YOLO11 model
model = YOLO(os.path.join(BaseAddress, "Gray-320-n.pt"))

# Export the model to TensorRT format
model.export(format="engine",
            imgsz=(640, 640),
            dynamic=False,
            #  int8=True,
            batch=1,
            #  half=True,
             verbose=False,
             simplify=True)
