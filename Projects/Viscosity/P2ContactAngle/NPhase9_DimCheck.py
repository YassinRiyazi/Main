import os
from PIL import Image

def check_image_health(root_dir, expected_height=130, expected_width=450):
    bad_images = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.jpg'):
                file_path = os.path.join(dirpath, filename)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        if height != expected_height or width > expected_width:
                            print(f"[WRONG SIZE] {file_path} -> {height}x{width}")
                            bad_images.append(file_path)
                except Exception as e:
                    print(f"[CORRUPTED] {file_path} -> {e}")
                    bad_images.append(file_path)


    print(f"\nChecked all images. Found {len(bad_images)} problematic images.")
    return bad_images

# Example usage
if __name__ == "__main__":
    # adress = "/media/d2u25/Dont/S4S-ROF/frame_Extracted"  # Replace with your folder path
    adress = r"/media/d2u25/Dont/S4S-ROF/frame_Extracted/300/S3-SNr3.05_D/frame_Extracted20250707_201500_DropNumber_11"
    check_image_health(adress)
