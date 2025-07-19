import os

def delete_empty_folders(root_dir):
    """
    Recursively deletes all empty folders under the given root directory.

    Args:
        root_dir (str): Path to the root directory.

    Returns:
        int: Number of folders deleted.
    """
    deleted_count = 0

    # Walk bottom-up to ensure we check subfolders before parents
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # If directory is empty (no files and no subdirectories)
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Deleted empty folder: {dirpath}")
                deleted_count += 1
            except OSError as e:
                print(f"Failed to delete {dirpath}: {e}")

    print(f"Total empty folders deleted: {deleted_count}")
    return deleted_count

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Delete all empty folders recursively.")
    # parser.add_argument("path", help="Root directory to clean up")

    # args = parser.parse_args()
    # delete_empty_folders(args.path)
    delete_empty_folders("frames")
    delete_empty_folders("frame_Extracted")

    