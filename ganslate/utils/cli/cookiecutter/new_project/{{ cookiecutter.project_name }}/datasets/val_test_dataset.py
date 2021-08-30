from pathlib import Path
from loguru import logger
import torch

from typing import Tuple
from dataclasses import dataclass

from torch.utils.data import Dataset
from omegaconf import MISSING

from ganslate import configs



@dataclass
class TemplateValTestDatasetConfig(configs.base.BaseDatasetConfig):
    # The name of the PyTorch dataset class defined below
    name: str = "CBCTtoCTValidationDataset"
    # Define other attributes, e.g.:
    patch_size = Tuple[int, int] = [128, 128]
    ...


class CBCTtoCTValTestDataset(Dataset):

    def __init__(self, conf):
        self.root_path = Path(conf[conf.mode].dataset.root).resolve()
        

    def __getitem__(self, index):
        # Assumes that root_path contains
        path_A = self.root_path[index] / "A.png"
        path_B = self.root_path[index] / "B.png"

        CBCT = sitk_utils.load(path_CBCT)
        CT = sitk_utils.load(path_CT)

        # Only if you need it in `save()` method as explained below
        metadata = {
            'path': str(path_CBCT),
            ...
        }

        CBCT, CT = clamp_normalize(CBCT, CT, self.hu_min, self.hu_max)

        return {'A': CBCT, 
                'B': CT, 
                # metadata [Optional] - if `save()` is defined and needs metadata
                # for saving the images in a different format during validation
                "metadata": metadata,
                # masks [Optional] - a dict of masks, used in validation to
                # calculate metrics in specific regions as well
                "masks": {"BODY": body_mask}
                }
    
     
    def save(self, tensor, save_dir, metadata):
        """Th
        """
        pass

    def __len__(self):
        return len(self.pairs)

