import os.path
import torch
import numpy as np
from torch.utils.data import Dataset
from midaGAN.utils.io import make_dataset_of_files


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
        shape = (1, 32, 32, 32)
        A = torch.rand(*shape)
        B = torch.rand(*shape)

        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.A_size, self.B_size)
