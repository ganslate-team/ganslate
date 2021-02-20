import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from midaGAN.utils.io import make_dataset_of_files
from midaGAN.utils import sitk_utils
from midaGAN.data.utils.normalization import z_score_normalize
from midaGAN.data.utils.stochastic_focal_patching import StochasticFocalPatchSampler

# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN import configs


@dataclass
class BratsValTestDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "BratsValTestDataset"
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


class BratsValTestDataset(Dataset):

    def __init__(self, conf):
        dir_brats = conf[conf.mode].dataset.root
        self.paths_brats = make_dataset_of_files(dir_brats, EXTENSIONS)
        self.num_datapoints = len(self.paths_brats)

        self.source_sequence = conf[conf.mode].dataset.source_sequence
        self.target_sequence = conf[conf.mode].dataset.target_sequence

    def __getitem__(self, index):
        mri = sitk_utils.load(self.paths_brats[index])

        A = get_mri_sequence(mri, self.source_sequence)
        B = get_mri_sequence(mri, self.target_sequence)

        A = sitk_utils.get_tensor(A)
        B = sitk_utils.get_tensor(B)

        # Z-score normalization per volume
        A = z_score_normalize(A, scale_to_range=(-1, 1))
        B = z_score_normalize(B, scale_to_range=(-1, 1))

        # Add channel dimension (1 = grayscale)
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)

        return {'A': A, 'B': B}

    def __len__(self):
        return self.num_datapoints

    def denormalize(self, tensor):
        """Allows the Tester and Validator to calculate the metrics in
        the original range of values.
        """
        # TODO: TEMPORARYYYYY
        return (tensor + 1) / 2
