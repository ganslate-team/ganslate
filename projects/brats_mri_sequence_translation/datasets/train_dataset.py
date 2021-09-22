import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from ganslate.utils.io import make_dataset_of_files
from ganslate.utils import sitk_utils
from ganslate.data.utils.normalization import z_score_normalize
from ganslate.data.utils.stochastic_focal_patching import StochasticFocalPatchSampler

# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from ganslate import configs


@dataclass
class BratsDatasetConfig(configs.base.BaseDatasetConfig):
    patch_size: Tuple[int, int, int] = (32, 32, 32)
    # Proportion of focal region size compared to original volume size
    focal_region_proportion: float = 0
    source_sequence: str = "flair"
    target_sequence: str = "t1w"


EXTENSIONS = ['.nii.gz']

# MRI sequences z-axis indices in Brats
SEQUENCE_MAP = {"flair": 0, "t1w": 1, "t1gd": 2, "t2w": 3}


def get_mri_sequence(sitk_image, sequence_name):
    z_index = SEQUENCE_MAP[sequence_name.lower()]

    size = list(sitk_image.GetSize())
    size[3] = 0
    index = [0, 0, 0, z_index]

    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(size)
    extractor.SetIndex(index)
    return extractor.Execute(sitk_image)


class BratsDataset(Dataset):

    def __init__(self, conf):
        dir_brats = conf.train.dataset.root
        self.paths_brats = make_dataset_of_files(dir_brats, EXTENSIONS)
        self.num_datapoints = len(self.paths_brats)

        focal_region_proportion = conf.train.dataset.focal_region_proportion
        self.patch_size = np.array(conf.train.dataset.patch_size)
        self.patch_sampler = StochasticFocalPatchSampler(self.patch_size, focal_region_proportion)

        self.source_sequence = conf.train.dataset.source_sequence
        self.target_sequence = conf.train.dataset.target_sequence

    def __getitem__(self, index):
        index_A = index % self.num_datapoints
        index_B = random.randint(0, self.num_datapoints - 1)

        path_A = self.paths_brats[index_A]
        path_B = self.paths_brats[index_B]

        # load nrrd as SimpleITK objects
        A = sitk_utils.load(path_A)
        B = sitk_utils.load(path_B)

        A = get_mri_sequence(A, self.source_sequence)
        B = get_mri_sequence(B, self.target_sequence)

        if (sitk_utils.is_image_smaller_than(A, self.patch_size) or
                sitk_utils.is_image_smaller_than(B, self.patch_size)):
            raise ValueError("Volume size not smaller than the defined patch size.\
                              \nA: {} \nB: {} \npatch_size: {}."\
                             .format(sitk_utils.get_torch_like_size(A),
                                     sitk_utils.get_torch_like_size(B),
                                     self.patch_size))

        A = sitk_utils.get_tensor(A)
        B = sitk_utils.get_tensor(B)

        # Extract patches
        A, B = self.patch_sampler.get_patch_pair(A, B)
        # Z-score normalization per volume
        A = z_score_normalize(A, scale_to_range=(-1, 1))
        B = z_score_normalize(B, scale_to_range=(-1, 1))

        # Add channel dimension (1 = grayscale)
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)

        return {'A': A, 'B': B}

    def __len__(self):
        return self.num_datapoints
