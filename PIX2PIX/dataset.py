from PIL import Image 
import numpy as np
import os
from torch.utils.data import Dataset
import config

class SatDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        print(self.list_files)
        
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        
        #Since our image contains both the Input_Image and the Target_Image so we will slice the imamge from middle
        input_image = image[:, :600, :]  
        target_image = image[:, 600:, :]
        
        augmentations = config.both_transform(image = input_image, image0 = target_image)
        input_image , target_image = augmentations["image"], augmentations["image0"]
        input_image = config.transform_only_input(image = input_image)["image"]
        target_image = config.transform_only_mask(image= target_image)["image"]
        
        # augmentations = both_transform(image = input_image, image0 = target_image)
        # input_image , target_image = augmentations["image"], augmentations["image0"]
        # input_image = transform_only_input(image = input_image)["iamge"]
        # target_image = transform_only_mask(image = target_image)["image"]
        
        return input_image, target_image
        
        
        