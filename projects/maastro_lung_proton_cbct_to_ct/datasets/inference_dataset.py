from pathlib import Path
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

import midaGAN
from midaGAN.utils import sitk_utils
from midaGAN.data.utils.normalization import min_max_normalize, min_max_denormalize
from midaGAN.data.utils.body_mask import apply_body_mask

# Config imports
from typing import Tuple
from dataclasses import dataclass
from omegaconf import MISSING
from midaGAN import configs

logger = logging.getLogger(__name__)


@dataclass
class CBCTtoCTInferenceDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "CBCTtoCTInferenceDataset"
    hounsfield_units_range: Tuple[int, int] = (-1000, 2000)


class CBCTtoCTInferenceDataset(Dataset):

    def __init__(self, conf):
        self.root_path = Path(conf.infer.dataset.root).resolve()

        self.paths_CBCT = []
        for CT_CBCT_pair_dir in self.root_path.iterdir():
            CT_CBCT_pair_dir = self.root_path / CT_CBCT_pair_dir
            CBCT = list(CT_CBCT_pair_dir.rglob('CBCT.nrrd'))[0]
            self.paths_CBCT.append(CBCT)

        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.infer.dataset.hounsfield_units_range

    def __getitem__(self, index):
        path_CBCT = self.paths_CBCT[index]

        CBCT = sitk_utils.load(path_CBCT)

        metadata = {
            'path': str(path_CBCT),
            'size': CBCT.GetSize(),
            'origin': CBCT.GetOrigin(),
            'spacing': CBCT.GetSpacing(),
            'direction': CBCT.GetDirection(),
            'dtype': sitk_utils.get_npy_dtype(CBCT)
        }

        CBCT = apply_body_mask(sitk_utils.get_npy(CBCT),
                               apply_mask=True,
                               masking_value=self.hu_min,
                               hu_threshold=-800)

        CBCT = torch.tensor(CBCT)

        # Limits the lowest and highest HU unit
        CBCT = torch.clamp(CBCT, self.hu_min, self.hu_max)

        # Normalize Hounsfield units to range [-1,1]
        CBCT = min_max_normalize(CBCT, self.hu_min, self.hu_max)

        # Add channel dimension (1 = grayscale)
        CBCT = CBCT.unsqueeze(0)

        return {'input': CBCT, "metadata": metadata}

    def __len__(self):
        return len(self.paths_CBCT)

    def save(self, tensor, save_dir, metadata):
        tensor = tensor.squeeze().cpu()
        tensor = min_max_denormalize(tensor, self.hu_min, self.hu_max)

        sitk_image = sitk_utils.tensor_to_sitk_image(tensor, metadata['origin'],
                                                     metadata['spacing'], metadata['direction'],
                                                     metadata['dtype'])

        # Dataset used has a directory per each datapoint, the name of each
        # datapoint's dir is used to save the output
        datapoint_path = Path(str(metadata['path']))

        save_path = datapoint_path.relative_to(self.root_path)

        save_path = Path(save_dir) / save_path

        save_path.parent.mkdir(exist_ok=True, parents=True)

        sitk_utils.write(sitk_image, save_path)
