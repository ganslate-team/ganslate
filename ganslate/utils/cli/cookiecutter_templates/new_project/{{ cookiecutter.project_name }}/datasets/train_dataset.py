import random
from pathlib import Path
import torch

from typing import Tuple
from dataclasses import dataclass

from torch.utils.data import Dataset
from omegaconf import MISSING

from ganslate import configs


@dataclass
class {{cookiecutter.dataset_name}}TrainConfig(configs.base.BaseDatasetConfig):
    # Define other attributes, e.g.:
    patch_size: Tuple[int, int] = [128, 128]
    ...


class {{cookiecutter.dataset_name}}TrainDataset(Dataset):

    def __init__(self, conf):
        root_path = Path(conf.train.dataset.root).resolve()

        # Assumes `A` and `B` dirs only for demonstration
        self.paths_A = root_path / "A"
        self.paths_B = root_path / "B"

        self.num_datapoints_A = len(self.paths_A)
        self.num_datapoints_B = len(self.paths_B)
        ...

    def __getitem__(self, index):
        # Get the pair A and B.
        # In unpaired training, select a random index for 
        # image B so that A and B pairs are not always the same.
        # For paired training, it depends on how the data is structured.
        index_A = index % self.num_datapoints_A
        index_B = random.randint(0, self.num_datapoints_B - 1)

        path_A = self.paths_A[index_A]
        path_B = self.paths_B[index_B]

        # Read the images, `read` is a placeholder
        A = read(path_A)
        B = read(path_B)

        # Preprocess and normalize to [-1,1], `preprocess` is a placeholder
        A = preprocess(A)
        B = preprocess(B)

        # You need to return a dict with `A` and `B` entries
        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.num_datapoints_A, self.num_datapoints_B)
