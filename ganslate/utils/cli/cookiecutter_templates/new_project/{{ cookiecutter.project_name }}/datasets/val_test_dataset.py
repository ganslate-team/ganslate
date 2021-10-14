from pathlib import Path
import torch

from typing import Tuple
from dataclasses import dataclass

from torch.utils.data import Dataset
from omegaconf import MISSING

from ganslate import configs


@dataclass
class {{cookiecutter.dataset_name}}ValTestDatasetConfig(configs.base.BaseDatasetConfig):
    # Define other attributes, e.g.:
    patch_size: Tuple[int, int] = [128, 128]
    ...


class {{cookiecutter.dataset_name}}ValTestDataset(Dataset):

    def __init__(self, conf):
        self.root_path = Path(conf[conf.mode].dataset.root).resolve()
        
    def __getitem__(self, index):
        # Assigning paths for A and B depends on your dataset dir structure
        path_A = self.root_path[index] / "A.png"
        path_B = self.root_path[index] / "B.png"

        # Read the images, `read` is a placeholder
        A = read(path_A)
        B = read(path_B)

        # Preprocess and normalize to [-1,1], `preprocess` is a placeholder
        A = preprocess(A)
        B = preprocess(B)

        # Metadata is optionally returned by this method, explained at the end of the method.
        # Delete if not necessary.
        metadata = {
            'path': str(path_A),
            ...
        }

        # Masks are optionally returned by this method, explained at the end of the method.
        # Delete if not necessary.
        masks = {}
        path_foreground_mask = self.root_path[index] / "foreground.png"
        foreground_mask = read(path_mask)
        masks["foreground"] = foreground_mask

        return {'A': A, 
                'B': B, 
                # [Optional] metadata - if `save()` is defined *and* if it requires metadata.
                "metadata": metadata,
                # [Optional] masks - a dict of masks, used during the validation or
                # testing to also calculate metrics over specific regions of the image.
                "masks": masks
                }
     
    def save(self, tensor, save_dir, metadata=None):
        """ By default, ganslate logs images in png format. However, if you wish
        to save images in a different way, then implement this `save()` method. 
        For example, you could save medical images in their native format for easier
        inspection or usage.
        If you do not need this method, remove it.
        """
        pass

    def __len__(self):
        # Depending on the dataset dir structure, you might want to change it.
        return len(self.root_path)

