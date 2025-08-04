"""
Author: Sajjad Shumaly
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def drop_cropping(
    diff_img: np.ndarray,
    drop_middle_y: int,
    drop_start: int,
    x_left_margin: int = 30,
    x_right_margin: int = 60,
    y_up_margin: int = 10,
    object_detection_threshold: int = 40,
) -> tuple[np.ndarray, int, int, int, int]:
    """
    Crops the droplet reflection from the difference image based on detected edges.

    Args:
        diff_img (np.ndarray): Difference image containing the droplet reflection.
        drop_middle_y (int): Approximate vertical center of the droplet.
        drop_start (int): Approximate starting point of the droplet.
        x_left_margin (int): Margin added to the left crop boundary.
        x_right_margin (int): Margin added to the right crop boundary.
        y_up_margin (int): Margin subtracted from the upper crop boundary.
        object_detection_threshold (int): Pixel intensity threshold for detecting droplet edges.

    Returns:
        tuple:
            - Cropped droplet reflection image (np.ndarray)
            - x_left (int): Left x-coordinate of the crop.
            - x_right (int): Right x-coordinate of the crop.
            - y_up (int): Top y-coordinate of the crop.
            - y_down (int): Bottom y-coordinate of the crop.

    Raises:
        ValueError: If the input image is not a valid numpy array.
    """
    if not isinstance(diff_img, np.ndarray):
        raise ValueError("Input image must be a valid numpy array.")

    height, width = diff_img.shape[:2]

    # Find left boundary
    left_candidates = diff_img[drop_middle_y, :, 0] > object_detection_threshold
    x_left = np.argmax(left_candidates) - x_left_margin
    x_left = max(0, x_left)

    # Find right boundary
    right_candidates = diff_img[drop_middle_y, ::-1, 0] > object_detection_threshold
    x_right = width - np.argmax(right_candidates) + x_right_margin
    x_right = min(width, x_right)

    # Find bottom boundary
    y_down = drop_middle_y
    for y in range(drop_middle_y, drop_start):
        row = diff_img[y, x_left:width, 0] > object_detection_threshold
        if np.any(row):
            y_down = y
            break

    # Find top boundary
    y_up = drop_middle_y
    for y in range(drop_middle_y, 0, -1):
        row = diff_img[y, x_left:width, 0] > object_detection_threshold
        if np.any(row):
            y_up = y
            break
    y_up = max(0, y_up - y_up_margin)

    # Crop and return
    drop_reflection = diff_img[y_up:y_down, x_left:x_right]
    return drop_reflection, x_left, x_right, y_up, y_down


if __name__ == "__main__":
    middle_drop_height = 25
    drop_start_height = 9
    baseline = 128
    drop_start = baseline - drop_start_height
    drop_middle_y = drop_start - middle_drop_height

    diff_img = cv2.imread("src/PyThon/ContactAngle/DropDetection/doc/002667.jpg")
    diff_img = diff_img[:100, :, :]  # Crop to baseline height
    drop_reflection, x_left, x_right, y_up, y_down = drop_cropping(
        diff_img,
        drop_middle_y=drop_middle_y,
        drop_start=drop_start,
        x_left_margin=30,
        x_right_margin=60,
        y_up_margin=10,
    )

    just_drop = diff_img[y_up:baseline, x_left:x_right, :]
    plt.imshow(just_drop)
    plt.show()
