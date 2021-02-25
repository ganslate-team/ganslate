import random
from dataclasses import dataclass, field
from pathlib import Path
# Config imports
from typing import Tuple

import midaGAN
import numpy as np
import torch
from midaGAN import configs
from midaGAN.data.utils.body_mask import apply_body_mask, get_body_mask
from midaGAN.data.utils.fov_truncate import truncate_CBCT_based_on_fov
from midaGAN.data.utils.normalization import (min_max_denormalize, min_max_normalize)
from midaGAN.data.utils.ops import pad
from midaGAN.data.utils.registration_methods import (register_CT_to_CBCT,
                                                     truncate_CT_to_scope_of_CBCT)
from midaGAN.data.utils.stochastic_focal_patching import \
    StochasticFocalPatchSampler
from midaGAN.utils import sitk_utils
from midaGAN.utils.io import load_json, make_recursive_dataset_of_files
from omegaconf import MISSING
from torch.utils.data import Dataset

DEBUG = False

import logging

logger = logging.getLogger(__name__)

EXTENSIONS = ['.nrrd']

# --------------------------- INFERENCE DATASET ----------------------------------------------
# --------------------------------------------------------------------------------------------


@dataclass
class CBCTtoCTInferenceDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "CBCTtoCTInferenceDataset"
    hounsfield_units_range: Tuple[int, int] = field(default_factory=lambda: (-1000, 2000))
    enable_masking: bool = False
    cbct_mask_threshold: int = -700


class CBCTtoCTInferenceDataset(Dataset):
    def __init__(self, conf):
        self.root_path = Path(conf.infer.dataset.root).resolve()

        self.paths = []

        for patient in self.root_path.iterdir():
            self.paths.extend(make_recursive_dataset_of_files(patient / "CBCT", EXTENSIONS))

        self.num_datapoints = len(self.paths)
        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.infer.dataset.hounsfield_units_range

        self.apply_mask = conf.infer.dataset.enable_masking
        self.cbct_mask_threshold = conf.infer.dataset.cbct_mask_threshold

    def __getitem__(self, index):
        path = str(self.paths[index])

        # load nrrd as SimpleITK objects
        volume = sitk_utils.load(path)

        volume = volume - 1024

        volume = truncate_CBCT_based_on_fov(volume)

        metadata = {
            'path': str(path),
            'size': volume.GetSize(),
            'origin': volume.GetOrigin(),
            'spacing': volume.GetSpacing(),
            'direction': volume.GetDirection(),
            'dtype': sitk_utils.get_npy_dtype(volume)
        }

        volume = sitk_utils.get_npy(volume)

        volume = apply_body_mask(volume, apply_mask=self.apply_mask, \
                                    hu_threshold=self.cbct_mask_threshold)
        volume = torch.tensor(volume)
        # Limits the lowest and highest HU unit
        volume = torch.clamp(volume, self.hu_min, self.hu_max)
        # Normalize Hounsfield units to range [-1,1]
        volume = min_max_normalize(volume, self.hu_min, self.hu_max)
        # Add channel dimension (1 = grayscale)
        volume = volume.unsqueeze(0)
        return volume, metadata

    def __len__(self):
        return self.num_datapoints

    def save(self, tensor, save_dir, metadata=None):
        tensor = tensor.squeeze().cpu()
        tensor = min_max_denormalize(tensor, self.hu_min, self.hu_max)

        if metadata:
            sitk_image = sitk_utils.tensor_to_sitk_image(tensor, metadata['origin'],
                                                         metadata['spacing'], metadata['direction'],
                                                         metadata['dtype'])

            datapoint_path = Path(str(metadata['path']))
            save_path = datapoint_path.relative_to(self.root_path)

        else:
            sitk_image = sitk_utils.tensor_to_sitk_image(tensor)
            save_path = f'image_{date.today().strftime("%b-%d-%Y")}.nrrd'

        # Dataset used has a directory per each datapoint, the name of each datapoint's dir is used to save the output
        save_path = Path(save_dir) / save_path
        save_path.parent.mkdir(exist_ok=True, parents=True)
        sitk_utils.write(sitk_image, save_path)
