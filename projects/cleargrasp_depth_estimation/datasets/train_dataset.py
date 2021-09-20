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

from ganslate.data.utils.normalization import min_max_normalize


EXTENSIONS = ['.jpg', '.exr']


# Max allowed intenity of depthmap images. Specified in metres. 
# This value is chosen by analyzing max values throughout the dataset.  
UPPER_DEPTH_INTENSITY_LIMIT = 8.0  



@dataclass
class ClearGraspTrainDatasetConfig(configs.base.BaseDatasetConfig):
    load_size: Tuple[int, int] = (512, 256)
    paired: bool = True   # `True` for paired A-B. 
    require_domain_B_rgb: bool = False  # Whether to fetch noisy RGB photo for domain B


class ClearGraspTrainDataset(Dataset):
    """
    Multimodality dataset containing RGB photos, surface normalmaps and depthmaps.
    Curated from Cleargrasp robot-vision dataset.
    The domain translation task is:   RGB + Normalmap --> Depthmap 
    """
    def __init__(self, conf):
        
        # self.mode = conf.mode
        self.paired = conf[conf.mode].dataset.paired 
        self.require_domain_B_rgb = conf[conf.mode].dataset.require_domain_B_rgb 

        rgb_dir = Path(conf[conf.mode].dataset.root) / "rgb"        
        normalmap_dir = Path(conf[conf.mode].dataset.root) / "normal"
        depthmap_dir = Path(conf[conf.mode].dataset.root) / "depth"

        self.image_paths = {'RGB': [], 'normalmap': [], 'depthmap': []}
        self.image_paths['RGB'] = make_dataset_of_files(rgb_dir, EXTENSIONS)
        self.image_paths['normalmap'] = make_dataset_of_files(normalmap_dir, EXTENSIONS)
        self.image_paths['depthmap'] = make_dataset_of_files(depthmap_dir, EXTENSIONS)
        self.dataset_size = len(self.image_paths['RGB'])

        self.load_size = conf[conf.mode].dataset.load_size
        self.load_resize_transform = transforms.Resize(
            size=(self.load_size[1], self.load_size[0]), interpolation=transforms.InterpolationMode.BICUBIC
            )
        
        # Clipping ranges
        self.rgb_min, self.rgb_max = 0.0, 255.0
        self.normalmap_min, self.normalmap_max = -1.0, 1.0
        self.depthmap_min, self.depthmap_max = 0.0, UPPER_DEPTH_INTENSITY_LIMIT


    def __len__(self):
        return self.dataset_size


    def __getitem__(self, index):

        # ------------
        # Fetch images

        index_A = index % self.dataset_size
        index_B = index_A if self.paired else random.randint(0, self.dataset_size - 1)
        index_A, index_B = 9, 1  ##
        
        image_path_A, image_path_B = {}, {}
        image_path_A['RGB'] = self.image_paths['RGB'][index_A]
        image_path_A['normalmap'] = self.image_paths['normalmap'][index_A]
        image_path_B['depthmap'] = self.image_paths['depthmap'][index_B]
        if self.require_domain_B_rgb:
            image_path_B['RGB'] = self.image_paths['RGB'][index_B]

        images_A, images_B = {}, {}
        images_A['RGB'] = read_rgb_to_tensor(image_path_A['RGB'])
        images_A['normalmap'] = read_normalmap_to_tensor(image_path_A['normalmap'])
        images_B['depthmap'] = read_depthmap_to_tensor(image_path_B['depthmap'])
        if self.require_domain_B_rgb:
            images_B['RGB'] = read_rgb_to_tensor(image_path_B['RGB'])


        # ------
        # Resize

        for k in images_A.keys():
            images_A[k] = self.load_resize_transform(images_A[k])
        for k in images_B.keys():
            images_B[k] = self.load_resize_transform(images_B[k])
        

        # ---------
        # Transform

        images_A, images_B = self.apply_transforms(images_A, images_B)


        # -------------
        # Normalization

        # Clip and then rescale all intensties to range [-1, 1]
        # Normalmap is already in this scale.
        images_A['RGB'] = clip_and_min_max_normalize(images_A['RGB'], self.rgb_min, self.rgb_max)
        images_A['normalmap'] = torch.clamp(images_A['normalmap'], self.normalmap_min, self.normalmap_max)
        images_B['depthmap'] = clip_and_min_max_normalize(images_B['depthmap'], self.depthmap_min, self.depthmap_max)
        if self.require_domain_B_rgb:
            images_B['RGB'] = clip_and_min_max_normalize(images_B['RGB'], self.rgb_min, self.rgb_max)


        # -------------------------
        # Add noise in domain-B RGB 

        if self.require_domain_B_rgb:
            images_B['RGB'] = images_B['RGB'] + torch.normal(mean=0, std=0.05, size=(self.load_size[1], self.load_size[0]))
            images_B['RGB'] = torch.clamp(images_B['RGB'], -1, 1)  # Clip to remove out-of-range overshoots


        # ---------------------
        # Construct sample dict 

        # A and B need to have dims (C,D,H,W)
        A = torch.cat([images_A['RGB'], images_A['normalmap']], dim=0)
        if self.require_domain_B_rgb:
            B = torch.cat([images_B['RGB'], images_B['depthmap']], dim=0)  
        else:
            B = images_B['depthmap']

        sample_dict = {'A': A, 'B': B}

        return sample_dict



    def apply_transforms(self, images_A, images_B):
        """
        TODO: What transform to use for augmentation? 
        Cannot naively apply random flip and crop, would mess up the normalmap and depthmap info, resp.
        Maybe flipping + changing normalmap colour mapping (by changing order of its RGB channels)
        """
        return images_A, images_B




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


def clip_and_min_max_normalize(tensor, min_value, max_value):
    tensor = torch.clamp(tensor, min_value, max_value)
    tensor = min_max_normalize(tensor, min_value, max_value)
    return tensor
