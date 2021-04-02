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

from midaGAN.utils.io import make_dataset_of_files

# Config imports
from dataclasses import dataclass
from midaGAN import configs

EXTENSIONS = ['.jpg', '.exr']

# Max allowed intenity of depthmap images. Specified in metres.
# This value is chosen by analyzing max values throughout the dataset.
UPPER_DEPTH_INTENSITY_LIMIT = 8.0


@dataclass
class ClearGraspCycleGANDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "ClearGraspCycleGANDataset"
    load_size: Tuple[int, int] = (512, 256)
    paired: bool = False  # `True` for paired A-B


class ClearGraspCycleGANDataset(Dataset):
    """
    Multimodality dataset containing RGB photos, surface normalmaps and depthmaps.
    Curated from Cleargrasp robot-vision dataset.
    Here, the GAN translation task is:   RGB + Normalmap --> Depthmap 
    This is the CycleGAN version:  
        Domain A:  RGB and Normalmap
        Domain B:  RGB and Depthmap  
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
            size=(load_size[1], load_size[0]), interpolation=transforms.InterpolationMode.BICUBIC)

    def __getitem__(self, index):
        index_A = index % self.dataset_size
        index_B = index_A if self.paired else random.randint(0, self.dataset_size - 1)

        rgb_A_path = self.rgb_paths[index_A]
        normal_path = self.normal_paths[index_A]
        rgb_B_path = self.rgb_paths[index_B]
        depth_path = self.depth_paths[index_B]

        rgb_A = read_rgb_to_tensor(rgb_A_path)
        normalmap = read_normalmap_to_tensor(normal_path)
        rgb_B = read_rgb_to_tensor(rgb_B_path)
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

        return {'A1': rgb_A, 'A2': normalmap, 'B1': rgb_B, 'B2': depthmap}

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
        Scale to [-1,1] range
        Normalmap already in this range, 
        but still rescale it to avoid offshoots caused by interpolation 
        """
        # Get limits, taking into account out-of-range offshoot values
        rgb_A_min, rgb_A_max = min(0.0, rgb_A.min()), max(255.0, rgb_A.max())
        normalmap_min, normalmap_max = min(-1.0, normalmap.min()), max(1.0, normalmap.max())
        rgb_B_min, rgb_B_max = min(0.0, rgb_B.min()), max(255.0, rgb_B.max())
        depthmap_min, depthmap_max = min(0.0, depthmap.min()), max(UPPER_DEPTH_INTENSITY_LIMIT,
                                                                   depthmap.max())

        rgb_A = (rgb_A - rgb_A_min) / (rgb_A_max - rgb_A_min) * 2 - 1
        normalmap = (normalmap - normalmap_min) / (normalmap_max - normalmap_min) * 2 - 1
        rgb_B = (rgb_B - rgb_B_min) / (rgb_B_max - rgb_B_min) * 2 - 1
        depthmap = (depthmap - depthmap_min) / (depthmap_max - depthmap_min) * 2 - 1
        return rgb_A, normalmap, rgb_B, depthmap


def read_rgb_to_tensor(path):
    """
    RGB reader based on cv2.imread(). 
    Just for consistency with normalmap and depthmap readers.
    """
    bgr_img = cv2.imread(path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img.transpose(2, 0, 1)  # (H,W,C) to (C,H,W) format
    return torch.tensor(rgb_img, dtype=torch.float32)


def read_normalmap_to_tensor(path):
    """
    Read normalmap image from EXR format to tensor of form (3,H,W) 
    """
    normalmap = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    normalmap = cv2.cvtColor(normalmap, cv2.COLOR_BGR2RGB)
    normalmap = normalmap.transpose(2, 0, 1)  # (H,W,C) to (C,H,W) format
    return torch.tensor(normalmap, dtype=torch.float32)


def read_depthmap_to_tensor(path):
    """
    Read depthmap image from EXR format to tensor of form (1,H,W) 
    """
    depthmap = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    depthmap = np.expand_dims(depthmap, axis=0)  # (H,W) to (1,H,W)
    return torch.tensor(depthmap, dtype=torch.float32)
