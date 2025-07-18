import cv2

def crop_and_save_image(input_path, output_path, x1, y1, x2, y2):
    """
    Crops a region from the input image and saves it to the output path.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the cropped image.
        x1 (int): Top-left x-coordinate.
        y1 (int): Top-left y-coordinate.
        x2 (int): Bottom-right x-coordinate.
        y2 (int): Bottom-right y-coordinate.

    Raises:
        ValueError: If the crop coordinates are invalid or image cannot be loaded.

    Example:
        crop_and_save_image("input.jpg", "output.jpg", 10, 20, 100, 200)
    """
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")

    height, width = image.shape[:2]

    # Clamp coordinates to image dimensions
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid crop coordinates: x1 >= x2 or y1 >= y2")

    cropped = image[y1:y2, x1:x2]
    cropped = cv2.bitwise_not(cropped)
    cv2.imwrite(output_path, cropped)
