import random
from pathlib import Path
from typing import Tuple

import glob
import numpy as np
import cv2
import PIL
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from midaGAN.utils.io import make_dataset_of_files

# Config imports
from dataclasses import dataclass
from midaGAN import configs


EXTENSIONS = ['.jpg', '.exr']


# Max allowed intenity of depthmap images. Specified in metres. 
# This value is chosen by analyzing max values throughout the dataset.  
UPPER_DEPTH_INTENSITY_LIMIT = 8.0  



@dataclass
class ClearGraspPix2PixDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "ClearGraspPix2PixDataset"
    load_size: Tuple[int, int] = (512, 256)
    paired: bool = True   # `True` for paired A-B  


class ClearGraspPix2PixDataset(Dataset):
    """
    Multimodality dataset containing RGB photos, surface normalmaps and depthmaps.
    Curated from Cleargrasp robot-vision dataset.
    Here, the GAN translation task is:   RGB + Normalmap --> Depthmap  
    This is the Pix2Pix version:  
        Domain A:  RGB and Normalmap
        Domain B:  Depthmap  
    """
    def __init__(self, conf):
        
        self.mode = conf.mode

        self.dir_rgb = Path(conf[conf.mode].dataset.root) / "rgb"        
        self.dir_normal = Path(conf[conf.mode].dataset.root) / "normal"
        self.dir_depth = Path(conf[conf.mode].dataset.root) / "depth"

        self.rgb_paths = make_dataset_of_files(self.dir_rgb, EXTENSIONS)
        self.normal_paths = make_dataset_of_files(self.dir_normal, EXTENSIONS)
        self.depth_paths = make_dataset_of_files(self.dir_depth, EXTENSIONS)
        self.dataset_size = len(self.rgb_paths)

        self.load_size = conf[conf.mode].dataset.load_size
        self.load_resize_transform = transforms.Resize(size=(self.load_size[1], self.load_size[0]), 
            interpolation=transforms.InterpolationMode.BICUBIC)
        

    def __getitem__(self, index):
        index_A = index % self.dataset_size
        index_B = index_A

        rgb_path = self.rgb_paths[index_A]
        normal_path = self.normal_paths[index_A]        
        depth_path = self.depth_paths[index_B]

        rgb_img = read_rgb_to_tensor(rgb_path)
        normalmap = read_normalmap_to_tensor(normal_path)
        depthmap = read_depthmap_to_tensor(depth_path)
        
        # Resize
        rgb_img = self.load_resize_transform(rgb_img)
        normalmap = self.load_resize_transform(normalmap)
        depthmap = self.load_resize_transform(depthmap)

        # Transforms
        rgb, normalmap, depthmap = self.apply_transforms(rgb_img, normalmap, depthmap)

        # Normalize
        rgb_img, normalmap, depthmap = self.normalize(rgb_img, normalmap, depthmap)

        # Make 3-channel image from 1-channel
        # depthmap = torch.cat([depthmap,depthmap,depthmap], dim=0)

        return {'A': torch.cat([rgb_img, normalmap], dim=0),  'B': depthmap}


    def __len__(self):
        return self.dataset_size


    def apply_transforms(self, rgb_img, normalmap, depthmap):
        """
        TODO: What transform to use for augmentation? 
        Cannot naively apply random flip and crop, would mess up the normalmap and depthmap info, resp.
        Maybe flipping + changing normalmap colour mapping (by changing order of its RGB channels)
        """
        return rgb_img, normalmap, depthmap


    def normalize(self, rgb_img, normalmap, depthmap):
        """
        Scale intensities to [-1,1] range
        Normalmap already in this range
        """
        # Ranges
        rgb_min, rgb_max = 0.0, 255.0
        depthmap_min, depthmap_max = 0.0, UPPER_DEPTH_INTENSITY_LIMIT
        
        # Normalize
        rgb_img = (rgb_img-rgb_min)/(rgb_max-rgb_min) * 2 - 1
        depthmap = (depthmap-depthmap_min)/(depthmap_max-depthmap_min) * 2 - 1
        return torch.clamp(rgb_img, -1, 1), torch.clamp(normalmap, -1, 1), torch.clamp(depthmap, -1, 1) 



def read_rgb_to_tensor(path):
    """
    RGB reader based on cv2.imread(). 
    Just for consistency with normalmap and depthmap readers.
    """
    bgr_img = cv2.imread(str(path))
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img.transpose(2,0,1)  # (H,W,C) to (C,H,W) format
    return torch.tensor(rgb_img, dtype=torch.float32)

def read_normalmap_to_tensor(path):
    """
    Read normalmap image from EXR format to tensor of form (3,H,W) 
    """
    normalmap = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    normalmap = cv2.cvtColor(normalmap, cv2.COLOR_BGR2RGB)
    normalmap = normalmap.transpose(2,0,1)  # (H,W,C) to (C,H,W) format
    return torch.tensor(normalmap, dtype=torch.float32)

def read_depthmap_to_tensor(path):
    """
    Read depthmap image from EXR format to tensor of form (1,H,W) 
    """
    depthmap = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    depthmap = np.expand_dims(depthmap, axis=0)  # (H,W) to (1,H,W)
    return torch.tensor(depthmap, dtype=torch.float32)