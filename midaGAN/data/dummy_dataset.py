import os.path
import torch
import numpy as np
from torch.utils.data import Dataset
from midaGAN.utils.io import make_dataset_of_files
# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf.config import BaseDatasetConfig

DUMMY_PATCH_SIZE = [32, 32, 32]

@dataclass
class DummyDatasetConfig(BaseDatasetConfig):
    name:         str = "dummy"
    patch_size: Tuple[int, int, int] = field(default_factory=lambda: DUMMY_PATCH_SIZE)


class DummyDataset(Dataset):
    """Dummy dataset for quick testing purposes"""
    def __init__(self, conf):
        self.conf = conf
        self.root = conf.dataset.root
        self.dir_AB = os.path.join(conf.dataset.root)
        self.AB_paths = sorted(make_dataset_of_files(self.dir_AB))
        self.A_size = 4
        self.B_size = self.A_size

    def __getitem__(self, index):
        #shape = (1, 128, 128, 128)
        shape = (1, *DUMMY_PATCH_SIZE)
        A = torch.rand(*shape)
        B = torch.rand(*shape)

        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.A_size, self.B_size)
