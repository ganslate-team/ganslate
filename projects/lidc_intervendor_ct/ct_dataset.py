from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from midaGAN.utils.io import make_dataset_of_files, load_json
from midaGAN.data.utils.normalization import min_max_normalize
from midaGAN.data.utils.stochastic_focal_patching import StochasticFocalPatchSampler


# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf import BaseDatasetConfig


@dataclass
class CTDatasetConfig(BaseDatasetConfig):
    name:         str = "CTDataset"
    patch_size:   Tuple[int] = field(default_factory=lambda: (32, 32, 32))
    focal_region_proportion: float = 0.2    # Proportion of focal region size compared to original volume size


EXTENSIONS = ['.npy']

class CTDataset(Dataset):
    def __init__(self, conf):
        dir_A = Path(conf.dataset.root) / 'A'
        dir_B = Path(conf.dataset.root) / 'B'
        self.A_paths = make_dataset_of_files(dir_A, EXTENSIONS)
        self.B_paths = make_dataset_of_files(dir_B, EXTENSIONS)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # Dataset range of values information for normalization
        norm_A = Path(conf.dataset.root) / 'normalize_A.json'
        norm_B = Path(conf.dataset.root) / 'normalize_B.json'
        self.norm_A = load_json(norm_A)
        self.norm_B = load_json(norm_B)

        patch_size = conf.dataset.patch_size
        focal_region_proportion = conf.dataset.focal_region_proportion
        self.patch_sampler = StochasticFocalPatchSampler(patch_size, focal_region_proportion)

    def __getitem__(self, index):
        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)

        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]
        
        A = torch.Tensor(np.load(A_path))
        B = torch.Tensor(np.load(B_path))
        
        A, B = self.patch_sampler.get_patch_pair(A, B) # Extract patches

        # Normalize Hounsfield units to range [-1,1]
        A = min_max_normalize(A, self.norm_A["min"], self.norm_A["max"])
        B = min_max_normalize(B, self.norm_B["min"], self.norm_B["max"])

        # Add channel dimension (1 = grayscale)
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)

        return {'A': A, 'B': B}


    def __len__(self):
        return max(self.A_size, self.B_size)




