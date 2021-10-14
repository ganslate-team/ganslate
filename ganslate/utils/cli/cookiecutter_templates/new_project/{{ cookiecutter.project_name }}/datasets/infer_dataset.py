from pathlib import Path
import torch

from typing import Tuple
from dataclasses import dataclass

from torch.utils.data import Dataset
from omegaconf import MISSING

from ganslate import configs


@dataclass
class {{cookiecutter.dataset_name}}InferDatasetConfig(configs.base.BaseDatasetConfig):
    # Define other attributes, e.g.:
    patch_size: Tuple[int, int] = [128, 128]
    ...


class {{cookiecutter.dataset_name}}InferDataset(Dataset):

    def __init__(self, conf):
        self.root_path = Path(conf.infer.dataset.root).resolve()


    def __getitem__(self, index):
        # Depends on your dataset dir structure
        path_A = self.root_path[index] / "A.png"

        # Read the images, `read` is a placeholder
        A = read(path_A)

        # Preprocess and normalize to [-1,1], `preprocess` is a placeholder
        A = preprocess(A)

        # Metadata is optionally returned by this method, explained at the end of the method.
        # Delete if not necessary.
        metadata = {
            'path': str(path_A),
            ...
        }

        return {
            # Notice that the key for inference input is not "A"
            "input": A,
            # [Optional] metadata - if `save()` is defined *and* if it requires metadata.
            "metadata": metadata,
        }
     
    def __len__(self):
        # Depending on the dataset dir structure, you might want to change it.
        return len(self.root_path)

    def save(self, tensor, save_dir, metadata=None):
        """ By default, ganslate logs images in png format. However, if you wish
        to save images in a different way, then implement this `save()` method. 
        For example, you could save medical images in their native format for easier
        inspection or usage.
        If you do not need this method, remove it.
        """
        pass
