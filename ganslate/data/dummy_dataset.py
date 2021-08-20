import torch
from torch.utils.data import Dataset
# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from ganslate import configs

DUMMY_PATCH_SIZE = [32, 32, 32]


@dataclass
class DummyDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "DummyDataset"
    patch_size: Tuple[int, int, int] = field(default_factory=lambda: DUMMY_PATCH_SIZE)


class DummyDataset(Dataset):
    """Dummy dataset for quick testing purposes"""

    def __init__(self, conf):
        self.root = conf.train.dataset.root
        self.A_size = 4
        self.B_size = self.A_size

    def __getitem__(self, index):
        shape = (1, *DUMMY_PATCH_SIZE)
        A = torch.rand(*shape)
        B = torch.rand(*shape)

        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.A_size, self.B_size)
