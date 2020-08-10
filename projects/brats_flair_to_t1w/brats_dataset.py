import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from midaGAN.utils.io import make_dataset_of_files
from midaGAN.utils.normalization import z_score_normalize
from midaGAN.utils import sitk_utils
from midaGAN.datasets.utils.stochastic_focal_patching import StochasticFocalPatchSampler

# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf.config import BaseDatasetConfig


@dataclass
class BratsDatasetConfig(BaseDatasetConfig):
    name:         str = "brats"
    patch_size:   Tuple[int, int, int] = field(default_factory=lambda: (32, 32, 32))
    focal_region_proportion: float = 0    # Proportion of focal region size compared to original volume size


EXTENSIONS = ['.nii.gz']

# MRI sequences z-axis index in Brats
FLAIR = 0
T1W = 1
T1GD = 2
T2W = 3

def get_mri_sequence(sitk_image, z_index):
    size = list(sitk_image.GetSize())
    size[3] = 0
    index = [0, 0, 0, z_index]
    
    Extractor = sitk.ExtractImageFilter()
    Extractor.SetSize(size)
    Extractor.SetIndex(index)
    return Extractor.Execute(sitk_image)


class BratsDataset(Dataset):
    def __init__(self, conf):
        dir_brats = conf.dataset.root
        self.paths_brats = make_dataset_of_files(dir_brats, EXTENSIONS)
        self.num_datapoints = len(self.paths_brats)

        focal_region_proportion = conf.dataset.focal_region_proportion
        self.patch_size = np.array(conf.dataset.patch_size)
        self.patch_sampler = StochasticFocalPatchSampler(self.patch_size, focal_region_proportion)

    # TODO: move it somewhere else
    def _is_volume_smaller_than_patch(self, sitk_volume):
        volume_size = sitk_utils.get_size_zxy(sitk_volume)
        if (volume_size < self.patch_size).any():
            return True
        return False

    def __getitem__(self, index):
        index_A = index % self.num_datapoints
        index_B = random.randint(0, self.num_datapoints - 1)

        path_A = self.paths_brats[index_A]
        path_B = self.paths_brats[index_B]
        
        # load nrrd as SimpleITK objects
        A = sitk_utils.load(path_A)
        B = sitk_utils.load(path_B)

        A = get_mri_sequence(A, FLAIR)
        B = get_mri_sequence(B, T1W)

        # TODO: make a function
        if self._is_volume_smaller_than_patch(A) or self._is_volume_smaller_than_patch(B):
            raise ValueError("Volume size not smaller than the defined patch size.\
                              \nA: {} \nB: {} \npatch_size: {}."\
                             .format(sitk_utils.get_size_zxy(A),
                                     sitk_utils.get_size_zxy(B), 
                                     self.patch_size))

        A = sitk_utils.get_tensor(A)
        B = sitk_utils.get_tensor(B)

        # Extract patches
        A, B = self.patch_sampler.get_patch_pair(A, B) 
        # Z-score normalization per volume
        A = z_score_normalize(A, scale_to_range=(-1,1))
        B = z_score_normalize(B, scale_to_range=(-1,1))
        
        # Add channel dimension (1 = grayscale)
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)

        return {'A': A, 'B': B}

    def __len__(self):
        return self.num_datapoints
