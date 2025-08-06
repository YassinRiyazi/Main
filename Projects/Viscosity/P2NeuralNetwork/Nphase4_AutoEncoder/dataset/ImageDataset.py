import  os
import  glob
import  tqdm
import  yaml
import  pickle
import  torchvision.transforms  as      transforms
from    PIL                     import  Image
from    torch.utils.data        import  Dataset

class loc_ImageDataset(Dataset):
    """

    Description:
        Train an embedding-based LSTM for time series data.
        Custom Dataset for Image Loading and Preprocessing 
        Supports caching of {index: path} dictionaries using YAML or pickle.

    Custom Dataset for loading and preprocessing images for autoencoder training.
    Supports caching of {index: path} dictionaries using YAML or pickle.
    
    TODO:
    - Add sequence image loading
    - Add corresponding viscosity labels to a dictionary


    Author: Yassin Riyazi

    Date:
        - 06-08-2025

    TODO:
    - Add sequence image loading
    - Add corresponding viscosity labels to a dictionary
    """

    def __init__(self,
                 data_dir: str = "/media/d2u25/Dont/frames_Process_15_Patch",
                 extension: str = ".png",
                 skip: int = 1,
                 index_file: str = "image_index.pkl",
                 load_from_file: bool = True,
                 use_yaml: bool = False):
        """
        Author:
            - Yassin Riyazi

        Date:
            - 06-08-2025

        Args:
            data_dir (str): Directory containing the images.
            extension (str): File extension of images.
            skip (int): Skip factor for selecting images.
            index_file (str): Path to save/load cached index.
            load_from_file (bool): Whether to load cached index instead of re-scanning.
            use_yaml (bool): Save index as YAML (otherwise uses pickle).
        """
        self.data_dir   = data_dir
        self.index_file = os.path.join(data_dir, index_file)
        self.use_yaml   = use_yaml

        if load_from_file and os.path.exists(self.index_file):
            print(f"Loading cached index from {self.index_file}...")
            if self.use_yaml:
                with open(self.index_file, "r") as f:
                    self.image_dict = yaml.safe_load(f)
            else:
                with open(self.index_file, "rb") as f:
                    self.image_dict = pickle.load(f)
        else:
            print("Scanning directories for image files (this may take a while)...")
            image_files = glob.glob(os.path.join(data_dir, "*", "*", "*", "*" + extension))
            image_files = image_files[::skip]

            # Build dictionary: {index: path}
            self.image_dict = {i: path for i, path in enumerate(image_files)}

            print(f"Saving cached index to {self.index_file}...")
            if self.use_yaml:
                with open(self.index_file, "w") as f:
                    yaml.dump(self.image_dict, f)
            else:
                with open(self.index_file, "wb") as f:
                    pickle.dump(self.image_dict, f)

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((201, 201)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_dict)

    def __getitem__(self, idx):
        img_path = self.image_dict[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image, #img_path  # Return image and its path

    def dims(self,) -> tuple[tuple[int, int], tuple[int, int]]:

        min_width   = float( 'inf')
        min_height  = float( 'inf')
        max_width   = float('-inf')
        max_height  = float('-inf')

        for path in tqdm.tqdm(self.image_dict.values(), desc="Calculating dimensions"):
            try:
                with Image.open(path) as img:
                    width, height   = img.size
                    min_width       = min(min_width, width)
                    min_height      = min(min_height, height)
                    max_width       = max(max_width, width)
                    max_height      = max(max_height, height)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        return (min_width, min_height), (max_width, max_height)



if __name__ == "__main__":
    # # First run: scans and saves index
    # dataset = loc_ImageDataset(
    #     data_dir="/media/d2u25/Dont/frames_Process_15_Patch",
    #     skip=4,
    #     load_from_file=False,
    #     use_yaml=False  # Change to True for YAML
    # )

    # Subsequent run: loads from pickle or YAML instantly
    fast_dataset = loc_ImageDataset(
        data_dir="/media/d2u25/Dont/frames_Process_15_Patch",
        skip=4,
        load_from_file=True,
        use_yaml=False
    )

    print(f"Number of images: {len(fast_dataset)}")
    sample_image, path = fast_dataset[0]
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample image path: {path}")
    print(f"Image dimensions (min, max): {fast_dataset.dims()}")

