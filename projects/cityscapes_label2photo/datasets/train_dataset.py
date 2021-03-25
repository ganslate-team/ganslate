import random
from pathlib import Path
from typing import Tuple

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from midaGAN.utils.io import make_dataset_of_files

# Config imports
from dataclasses import dataclass
from midaGAN import configs


EXTENSIONS = ['.png']


@dataclass
class Label2PhotoTrainDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "Label2PhotoTrainDataset"
    load_size: Tuple[int, int] = (572, 286)
    crop_size: Tuple[int, int] = (512, 256)
    random_flip: bool = True
    random_crop: bool = True
    paired: bool = True   # `True` for paired training  
    masking: bool = True  # `True` to mask away the "void" objects in the photos


class Label2PhotoTrainDataset(Dataset):

    def __init__(self, conf):
      
        self.dir_A = Path(conf.train.dataset.root) / 'A_color'        
        self.dir_B = Path(conf.train.dataset.root) / 'B'

        self.A_paths = make_dataset_of_files(self.dir_A, EXTENSIONS)
        self.B_paths = make_dataset_of_files(self.dir_B, EXTENSIONS)
        self.dataset_size = len(self.A_paths)

        self.random_flip = conf.train.dataset.random_flip
        self.random_crop = conf.train.dataset.random_crop

        self.load_size = conf.train.dataset.load_size
        self.crop_size = conf.train.dataset.crop_size
        
        self.paired = conf.train.dataset.paired
        self.masking = conf.train.dataset.masking


    def __getitem__(self, index):
        index_A = index % self.dataset_size
        index_B = index_A if self.paired else random.randint(0, self.dataset_size - 1)
        
        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')    

        # Resize
        A_img = A_img.resize(self.load_size, resample=Image.NEAREST)
        B_img = B_img.resize(self.load_size, resample=Image.BICUBIC)

        # Transform
        A_img, B_img = self.apply_transform(A_img, B_img)
        
        # Create a mask of valid (i.e. non-void) regions from GT color images (domain A) 
        # and apply on the photos (domain B)
        # Void class labels are black in GT color images
        # (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)
        if self.paired and self.masking:        # Masking meaningful only if A and B are paired
            validity_mask = np.array(A_img, dtype=np.uint8).mean(axis=2) != 0            
        else:
            validity_mask = None

        A = self.normalize(A_img)
        B = self.normalize(B_img, validity_mask)
        
        return {'A': A, 'B': B}


    def __len__(self):
        return self.dataset_size


    def apply_transform(self, A_img, B_img):
        r = random.random()
        if self.random_crop and r < 0.66:  # Random crop
            load_width, load_height = self.load_size
            crop_width, crop_height = self.crop_size
            random_left = random.randint(0, load_width - crop_width -1)
            random_top = random.randint(0, load_height - crop_height -1)
            A_img = TF.crop(A_img, top=random_top, left=random_left, 
                            height=crop_height, width=crop_width)
            B_img = TF.crop(B_img, top=random_top, left=random_left,
                            height=crop_height, width=crop_width)
            if self.random_flip and r < 0.33: # Then, Random flip
                A_img = TF.hflip(A_img)
                B_img = TF.hflip(B_img)
            
        else:
            A_img = A_img.resize(self.crop_size, resample=Image.NEAREST)
            B_img = B_img.resize(self.crop_size, resample=Image.BICUBIC)

        return A_img, B_img


    def normalize(self, x_img, validity_mask=None):
        x = transforms.ToTensor()(x_img) # Convert to tensor, scales to [0,1] range internally 
        
        if validity_mask is not None:
            validity_mask = torch.tensor(validity_mask) 
        else: 
            validity_mask = torch.ones_like(x)
        
        x = x * validity_mask
        x = x * 2 - 1   # Scale to [-1,1] range
        return x 