import os
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

from ganslate.data.utils.normalization import min_max_normalize, min_max_denormalize


EXTENSIONS = ['.jpg', '.exr']


# Max allowed intenity of depthmap images. Specified in metres. 
# This value is chosen by analyzing max values throughout the dataset.  
UPPER_DEPTH_INTENSITY_LIMIT = 8.0  



@dataclass
class ClearGraspValTestDatasetConfig(configs.base.BaseDatasetConfig):
    """
    Note: Val dataset is paired, and does not supply RGB in domain-B
    """
    load_size: Tuple[int, int] = (512, 256)
    model_is_cyclegan_balanced: bool = False


class ClearGraspValTestDataset(Dataset):
    """
    Multimodality dataset containing RGB photos, surface normalmaps and depthmaps.
    Curated from Cleargrasp robot-vision dataset.
    The domain translation task is:   RGB + Normalmap --> Depthmap 
    """
    def __init__(self, conf):

        rgb_dir = Path(conf[conf.mode].dataset.root) / "rgb"        
        normalmap_dir = Path(conf[conf.mode].dataset.root) / "normal"
        depthmap_dir = Path(conf[conf.mode].dataset.root) / "depth"

        self.image_paths = {'RGB': [], 'normalmap': [], 'depthmap': []}
        self.image_paths['RGB'] = make_dataset_of_files(rgb_dir, EXTENSIONS)
        self.image_paths['normalmap'] = make_dataset_of_files(normalmap_dir, EXTENSIONS)
        self.image_paths['depthmap'] = make_dataset_of_files(depthmap_dir, EXTENSIONS)
        self.dataset_size = len(self.image_paths['RGB'])

        self.sample_ids = ['-'.join(str(path).split('/')[-1].split('.')[0].split('-')[:-1]) \
            for path in self.image_paths['RGB']]

        self.load_size = conf[conf.mode].dataset.load_size
        self.load_resize_transform = transforms.Resize(
            size=(self.load_size[1], self.load_size[0]), interpolation=transforms.InterpolationMode.BICUBIC
            )
        
        # Clipping ranges
        self.rgb_min, self.rgb_max = 0.0, 255.0
        self.normalmap_min, self.normalmap_max = -1.0, 1.0
        self.depthmap_min, self.depthmap_max = 0.0, UPPER_DEPTH_INTENSITY_LIMIT

        # Using Cyclegan-balanced (v3) ?
        self.model_is_cyclegan_balanced = conf[conf.mode].dataset.model_is_cyclegan_balanced


    def __len__(self):
        return self.dataset_size


    def __getitem__(self, index):

        # ------------
        # Fetch images
        
        image_path = {}
        image_path['RGB'] = self.image_paths['RGB'][index]
        image_path['normalmap'] = self.image_paths['normalmap'][index]
        image_path['depthmap'] = self.image_paths['depthmap'][index]

        images = {}
        images['RGB'] = read_rgb_to_tensor(image_path['RGB'])
        images['normalmap'] = read_normalmap_to_tensor(image_path['normalmap'])
        images['depthmap'] = read_depthmap_to_tensor(image_path['depthmap'])


        # Store the sample ID, need while saving the predicted image
        metadata = {
            'sample_id': self.sample_ids[index]
        }


        # ------
        # Resize

        for k in images.keys():
            images[k] = self.load_resize_transform(images[k])


        # -------------
        # Normalization

        # Clip and then rescale all intensties to range [-1, 1]
        # Normalmap is already in this scale.
        images['RGB'] = clip_and_min_max_normalize(images['RGB'], self.rgb_min, self.rgb_max)
        images['normalmap'] = torch.clamp(images['normalmap'], self.normalmap_min, self.normalmap_max)
        images['depthmap'] = clip_and_min_max_normalize(images['depthmap'], self.depthmap_min, self.depthmap_max)
       

        # ---------------------
        # Construct sample dict 

        # A and B need to have dims (C,D,H,W)
        A = torch.cat([images['RGB'], images['normalmap']], dim=0)
       
        if self.model_is_cyclegan_balanced:
            zeros_dummy = torch.zeros_like(images['RGB'])
            B = torch.cat([zeros_dummy, images['depthmap']], dim=0)  
        else:
            B = images['depthmap']

        sample_dict = {'A': A, 'B': B}

        # Include meta data
        sample_dict['metadata'] = metadata

        return sample_dict



    def denormalize(self, tensor):
        """Allows the Tester and Validator to calculate the metrics in
        the original range of values.
        `tensor` can be either the predicted or the ground truth depthmap image tensor
        """
        tensor = min_max_denormalize(tensor, self.depthmap_min, self.depthmap_max)
        return tensor


    def save(self, tensor, save_dir, metadata):
        """ Save predicted tensors as EXR
        """
        
        # If the model is CycleGAN-balanced, tensor is 4-channel with the 
        # last channel containing depthmap and first 3 channels containing a dummy array. 
        if self.model_is_cyclegan_balanced:  # Convert from (C,H,W) to (H,W) format
            tensor = tensor[3]  #  (4,H,W) -> (H,W)
        else:
            tensor = tensor.squeeze()  # (1,H,W) -> (H,W)

        # Rescale back to [self.depthmap_min, self.depthmap_max]
        tensor = min_max_denormalize(tensor.cpu(), self.depthmap_min, self.depthmap_max)

        # Write to file
        os.makedirs(save_dir, exist_ok=True)
        sample_id = metadata['sample_id']
        save_path = f"{save_dir}/{sample_id}.exr"
        write_depthmap_tensor_to_exr(tensor, save_path)



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


def write_depthmap_tensor_to_exr(depthmap, path):
    depthmap = depthmap.numpy()
    cv2.imwrite(path, depthmap, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])


def clip_and_min_max_normalize(tensor, min_value, max_value):
    tensor = torch.clamp(tensor, min_value, max_value)
    tensor = min_max_normalize(tensor, min_value, max_value)
    return tensor
