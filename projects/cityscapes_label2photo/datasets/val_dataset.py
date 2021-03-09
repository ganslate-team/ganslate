import random
from pathlib import Path
from typing import Tuple

import PIL
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from midaGAN.data.utils.transforms import get_transform
from midaGAN.utils.io import make_dataset_of_files

# Config imports
from dataclasses import dataclass
from midaGAN import configs


EXTENSIONS = ['.png']


@dataclass
class Label2PhotoValDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "Label2PhotoValDataset"
    load_size: Tuple[int, int] = (512, 256)
    masking: bool = True
    # No option for flip transform 
    # No option for paired/unpaired data - Validation images are always paired 


class Label2PhotoValDataset(Dataset):

    def __init__(self, conf):
           
        self.dir_A = Path(conf.val.dataset.root) / 'A_color'        
        self.dir_B = Path(conf.val.dataset.root) / 'B'

        self.A_paths = make_dataset_of_files(self.dir_A, EXTENSIONS)
        self.B_paths = make_dataset_of_files(self.dir_B, EXTENSIONS)
        self.dataset_size = len(self.A_paths)

        self.load_size = conf.val.dataset.load_size        


    def __getitem__(self, index):
        index = index % self.dataset_size

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Resize
        A_img = A_img.resize(self.load_size, resample=PIL.Image.NEAREST)
        B_img = B_img.resize(self.load_size, resample=PIL.Image.BICUBIC)

        # Create a mask of valid regions
        if self.masking: 
            validity_mask = np.array(A_img, dtype=np.uint8).mean(axis=2) != 0            
        else:
            validity_mask = None

        A = self.normalize(A_img)
        B = self.normalize(B_img, validity_mask)
        return {'A': A, 'B': B}


    def __len__(self):
        return self.dataset_size


    def normalize(self, x_img, validity_mask=None):
        x = transforms.ToTensor()(x_img) # Convert to tensor, scales to [0,1] range internally 
        
        if validity_mask is not None:
            validity_mask = torch.tensor(validity_mask) 
        else: 
            validity_mask = torch.ones_like(x)
        
        x = x * validity_mask
        x = x * 2 - 1   # Scale to [-1,1] range
        return x 