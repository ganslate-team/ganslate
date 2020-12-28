from pathlib import Path
import random
import numpy as np
import torch
import torchvision
import pandas as pd
from torch.utils.data import Dataset
from midaGAN.data.utils.normalization import z_score_normalize_with_precomputed_stats
from midaGAN.utils import sitk_utils, io

# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf import BaseDatasetConfig


@dataclass
class SliceBasedDatasetConfig(BaseDatasetConfig):
    name: str = "SliceBasedDataset"
    image_channels: int = 1  # Number of image channels (1 for grayscale, 3 for RGB)

    pad: bool = False
    pad_size: Tuple[int, int] = field(default_factory=lambda: ())

    crop: bool = False
    crop_size: Tuple[int, int] = field(default_factory=lambda: ())


class SliceBasedDataset(Dataset):

    def __init__(self, conf):
        self.dir_root = Path(conf.dataset.root)

        self.pad_size = None
        self.crop_size = None
        if conf.dataset.pad and conf.dataset.crop:
            raise ValueError("Please specify either pad only, crop only or none.")
        elif conf.dataset.pad:
            self.pad_size = conf.dataset.pad_size
        elif conf.dataset.crop:
            self.crop_size = conf.dataset.crop_size

        dataset_summary = pd.read_csv(Path(conf.dataset.root) / 'dataset_summary.csv')
        # Filter out rows by their domain
        self.domain_A_summary = dataset_summary[dataset_summary["volume_filename"].str.startswith(
            'A')]
        self.domain_B_summary = dataset_summary[dataset_summary["volume_filename"].str.startswith(
            'B')]

        self.num_datapoints_A = len(self.domain_A_summary)
        self.num_datapoints_B = len(self.domain_B_summary)

    def __getitem__(self, index):
        index_A = int(index % self.num_datapoints_A)
        index_B = random.randint(0, self.num_datapoints_B - 1)

        summary_A = self.domain_A_summary.iloc[index_A]
        summary_B = self.domain_B_summary.iloc[index_B]

        path_A = self.dir_root / summary_A["volume_filename"]
        path_B = self.dir_root / summary_B["volume_filename"]

        # load volume as SimpleITK object
        A = sitk_utils.load(path_A)
        B = sitk_utils.load(path_B)

        A = sitk_utils.get_tensor(A)
        B = sitk_utils.get_tensor(B)

        # Take the slice
        A = A[summary_A["slice"]]
        B = B[summary_B["slice"]]

        # Z-score normalization per volume
        mean_std_A = (summary_A["volume_mean"], summary_A["volume_std"])
        mean_std_B = (summary_B["volume_mean"], summary_B["volume_std"])
        min_max_A = (summary_A["volume_min"], summary_A["volume_max"])
        min_max_B = (summary_B["volume_min"], summary_B["volume_max"])

        A = z_score_normalize_with_precomputed_stats(A,
                                                     scale_to_range=(-1, 1),
                                                     mean_std=mean_std_A,
                                                     original_scale=min_max_A)
        B = z_score_normalize_with_precomputed_stats(B,
                                                     scale_to_range=(-1, 1),
                                                     mean_std=mean_std_B,
                                                     original_scale=min_max_B)

        if self.pad_size:
            A = pad_tensor_to_shape(A, output_shape=self.pad_size)
            B = pad_tensor_to_shape(B, output_shape=self.pad_size)
        elif self.crop_size:
            A = random_crop_tensor_to_shape(A, output_shape=self.crop_size)
            B = random_crop_tensor_to_shape(B, output_shape=self.crop_size)

        # Add channel dimension (1 = grayscale)
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)

        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.num_datapoints_A, self.num_datapoints_B)


# TODO: move these two
def pad_tensor_to_shape(tensor, output_shape):
    output_shape = torch.tensor(output_shape)
    input_shape = torch.tensor(tensor.shape)
    pad_needed = output_shape - input_shape
    pad_needed = pad_needed.float() / 2  # on each side
    # left, right, top, bottom. Using ceil and floor to ensure the specified output dimension.
    pad = [pad_needed[1].ceil(), pad_needed[1].floor(), pad_needed[0].ceil(), pad_needed[0].floor()]
    pad = [int(value) for value in pad]
    return torch.nn.functional.pad(input=tensor, pad=pad, mode='constant', value=-1)


def random_crop_tensor_to_shape(tensor, output_shape):
    output_shape = torch.tensor(output_shape)
    input_shape = torch.tensor(tensor.shape)
    start_region = input_shape - output_shape
    x = torch.randint(start_region[0] + 1, (1,))
    y = torch.randint(start_region[1] + 1, (1,))
    cropped_tensor = tensor[x:x + output_shape[0], y:y + output_shape[1]]

    return cropped_tensor
