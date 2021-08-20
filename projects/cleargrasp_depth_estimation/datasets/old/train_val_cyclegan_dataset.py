import random
from pathlib import Path
from typing import Tuple

import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from ganslate.utils.io import make_dataset_of_files

# Config imports
from dataclasses import dataclass
from ganslate import configs


EXTENSIONS = ['.jpg', '.exr']


# Max allowed intenity of depthmap images. Specified in metres. 
# This value is chosen by analyzing max values throughout the dataset.  
UPPER_DEPTH_INTENSITY_LIMIT = 8.0  



@dataclass
class ClearGraspCycleGANDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "ClearGraspCycleGANDataset"
    load_size: Tuple[int, int] = (512, 256)
    paired: bool = False   # `True` for paired A-B.  Need paired during validation
    fetch_rgb_b: bool = False  # Whether to fetch noisy RGB photo for domain B


class ClearGraspCycleGANDataset(Dataset):
    """
    Multimodality dataset containing RGB photos, surface normalmaps and depthmaps.
    Curated from Cleargrasp robot-vision dataset.
    Here, the GAN translation task is:   RGB + Normalmap --> Depthmap 
    This is the CycleGAN version:  
        Domain A:  RGB photo and Normalmap
        Domain B:  Several options -- (1) Depthmap (used with CycleGAN multimodal v1)
                                      (2) Noisy RGB photo and Depthmap (used with v2 and v3) 
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
        self.load_resize_transform = transforms.Resize(
            size=(self.load_size[1], self.load_size[0]), interpolation=transforms.InterpolationMode.BICUBIC
            )
        
        self.paired = conf[conf.mode].dataset.paired 
        self.fetch_rgb_b = conf[conf.mode].dataset.fetch_rgb_b 


    def __getitem__(self, index):
        index_A = index % self.dataset_size
        index_B = index_A if self.paired else random.randint(0, self.dataset_size - 1)
        
        rgb_A_path = self.rgb_paths[index_A]
        normal_path = self.normal_paths[index_A] 
        rgb_B_path = self.rgb_paths[index_B]  if self.fetch_rgb_b  else None
        depth_path = self.depth_paths[index_B]

        rgb_A = read_rgb_to_tensor(rgb_A_path)
        normalmap = read_normalmap_to_tensor(normal_path)
        rgb_B = read_rgb_to_tensor(rgb_B_path) if self.fetch_rgb_b else torch.zeros_like(rgb_A)
        depthmap = read_depthmap_to_tensor(depth_path)

        # Resize
        rgb_A = self.load_resize_transform(rgb_A)
        normalmap = self.load_resize_transform(normalmap)
        rgb_B = self.load_resize_transform(rgb_B)
        depthmap = self.load_resize_transform(depthmap)
        
        # Transform
        rgb_A, normalmap, rgb_B, depthmap = self.apply_transforms(rgb_A, normalmap, rgb_B, depthmap)

        # Normalize
        rgb_A, normalmap, rgb_B, depthmap = self.normalize(rgb_A, normalmap, rgb_B, depthmap)

        # Add noise in B's RGB photo
        if self.fetch_rgb_b:
            rgb_B = rgb_B + torch.normal(mean=0, std=0.05, size=(self.load_size[1], self.load_size[0]))
            rgb_B = torch.clamp(rgb_B, -1, 1)  # Clip to remove out-of-range values

        # Prepare A and B
        A = torch.cat([rgb_A, normalmap], dim=0)
        if self.fetch_rgb_b:
            B = torch.cat([rgb_B, depthmap], dim=0)  
        else:
            B = depthmap

        return {'A': A, 'B': B}



    def __len__(self):
        return self.dataset_size


    def apply_transforms(self, rgb_A, normalmap, rgb_B, depthmap):
        """
        TODO: What transform to use for augmentation? 
        Cannot naively apply random flip and crop, would mess up the normalmap and depthmap info, resp.
        Maybe flipping + changing normalmap colour mapping (by changing order of its RGB channels)
        """
        return rgb_A, normalmap, rgb_B, depthmap


    def normalize(self, rgb_A, normalmap, rgb_B, depthmap):
        """
        Scale intensities to [-1,1] range
        Normalmap already in this range
        """
        # Ranges
        rgb_min, rgb_max = 0.0, 255.0
        depthmap_min, depthmap_max = 0.0, UPPER_DEPTH_INTENSITY_LIMIT
        
        # Normalize
        rgb_A = (rgb_A - rgb_min) / (rgb_max - rgb_min) * 2 - 1
        if self.fetch_rgb_b:
            rgb_B = (rgb_B - rgb_min) / (rgb_max - rgb_min) * 2 - 1
        depthmap = (depthmap - depthmap_min) / (depthmap_max - depthmap_min) * 2 - 1
        
        # Clip to remove out-of-range overshoots
        rgb_A = torch.clamp(rgb_A, -1, 1)
        normalmap = torch.clamp(normalmap, -1, 1)
        if self.fetch_rgb_b:
            rgb_B = torch.clamp(rgb_B, -1, 1)
        depthmap = torch.clamp(depthmap, -1, 1)

        return rgb_A, normalmap, rgb_B, depthmap



def read_rgb_to_tensor(path):
    """
    RGB reader based on cv2.imread(). 
    Just for consistency with normalmap and depthmap readers.
    """
    bgr_img = cv2.imread(str(path))
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img.transpose(2,0,1)  # (H,W,C) to (C,H,W)
    return torch.tensor(rgb_img, dtype=torch.float32)

def read_normalmap_to_tensor(path):
    """
    Read normalmap image from EXR format to tensor of form (3,H,W) 
    """
    normalmap = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    normalmap = cv2.cvtColor(normalmap, cv2.COLOR_BGR2RGB)
    normalmap = normalmap.transpose(2,0,1)  # (H,W,C) to (C,H,W)
    return torch.tensor(normalmap, dtype=torch.float32)

def read_depthmap_to_tensor(path):
    """
    Read depthmap image from EXR format to tensor of form (1,H,W) 
    """
    depthmap = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    depthmap = np.expand_dims(depthmap, axis=0)  # (H,W) to (1,H,W)
    return torch.tensor(depthmap, dtype=torch.float32)

