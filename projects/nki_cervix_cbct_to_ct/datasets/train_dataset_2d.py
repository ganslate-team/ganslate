import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
# Config imports
from typing import Tuple

import midaGAN
import numpy as np
import torch
from midaGAN import configs
from midaGAN.data.utils.body_mask import apply_body_mask
from midaGAN.data.utils.fov_truncate import truncate_CBCT_based_on_fov
from midaGAN.data.utils.normalization import (min_max_denormalize,
                                              min_max_normalize)
from midaGAN.data.utils.ops import pad

from midaGAN.data.utils.registration_methods import register_CT_to_CBCT
from midaGAN.data.utils.stochastic_focal_patching import \
    StochasticFocalPatchSampler
from midaGAN.utils import sitk_utils
from midaGAN.utils.io import load_json, make_recursive_dataset_of_files
from omegaconf import MISSING
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

EXTENSIONS = ['.nrrd']
DEBUG = False


@dataclass
class CBCTtoCT2DDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "CBCTtoCT2DDataset"
    image_size: Tuple[int, int] = (256, 256)
    hounsfield_units_range: Tuple[int, int] = field(
        default_factory=lambda: (-1000, 2000))  #TODO: what should be the default range
    focal_region_proportion: float = 0.2  # Proportion of focal region size compared to original volume size
    enable_cache: bool = False
    image_channels: int = 1
    enable_masking: bool = False
    ct_mask_threshold: int = -300
    cbct_mask_threshold: int = -700


class CBCTtoCT2DDataset(Dataset):

    def __init__(self, conf):

        root_path = Path(conf.train.dataset.root).resolve()
        self.paths_CBCT = []
        self.paths_CT = []

        for patient in root_path.iterdir():
            patient_cbcts = make_recursive_dataset_of_files(
                patient / "CBCT", EXTENSIONS)
            
            patient_cts = make_recursive_dataset_of_files(patient / "CT", EXTENSIONS)
            patient_cts = [path for path in patient_cts if path.stem == "CT"]

            self.paths_CBCT.extend(patient_cbcts)
            self.paths_CT.extend(patient_cts)

        self.num_datapoints_CBCT = len(self.paths_CBCT)
        self.num_datapoints_CT = len(self.paths_CT)

        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.train.dataset.hounsfield_units_range

        focal_region_proportion = conf.train.dataset.focal_region_proportion
        self.patch_size = np.array(conf.train.dataset.image_size)
        self.slice_sampler = StochasticFocalPatchSampler(self.patch_size, focal_region_proportion)

        self.apply_mask = conf.train.dataset.enable_masking
        self.cbct_mask_threshold = conf.train.dataset.cbct_mask_threshold
        self.ct_mask_threshold = conf.train.dataset.ct_mask_threshold

    def __getitem__(self, index):
        index_CBCT = index % self.num_datapoints_CBCT
        index_CT = random.randint(0, self.num_datapoints_CT - 1)

        path_CBCT = self.paths_CBCT[index_CBCT]
        path_CT = self.paths_CT[index_CT]

        # load nrrd as SimpleITK objects
        CBCT = sitk_utils.load(path_CBCT)
        CT = sitk_utils.load(path_CT)

        # Subtract 1024 from CBCT to map values from grayscale to HU approx
        CBCT = CBCT - 1000

        # Truncate CBCT based on size of FOV in the image
        CBCT = truncate_CBCT_based_on_fov(CBCT)
        # Register CT to CBCT using rigid registration
        CT = register_CT_to_CBCT(CT, CBCT)

        # Mask and bound is applied on numpy arrays!
        CBCT = sitk_utils.get_npy(CBCT)
        CT = sitk_utils.get_npy(CT)

        # Apply body masking to the CT and CBCT arrays
        # and bound the z, x, y grid to around the mask
        try:
            CBCT = apply_body_mask(CBCT, \
                    apply_mask=self.apply_mask,  hu_threshold=self.cbct_mask_threshold)
        except:
            logger.error(f"Error applying mask and bound in file : {path_CBCT}")

        try:
            CT = apply_body_mask(CT, \
                    apply_mask=self.apply_mask, hu_threshold=self.ct_mask_threshold)

        except:
            logger.error(f"Error applying mask and bound in file : {path_CT}")

        CBCT = pad(CBCT, np.expand_dims(self.patch_size, axis=0))
        CT = pad(CT, np.expand_dims(self.patch_size, axis=0))

        if DEBUG:
            import wandb

            logdict = {
                "CBCT_3D": wandb.Image(CBCT[CBCT.shape[0] // 2], caption=str(path_CBCT)),
                "CT_3D": wandb.Image(CT[CT.shape[0] // 2], caption=str(path_CT))
            }
            wandb.log(logdict)

        if DEBUG:
            import wandb

            logdict = {
                "CBCT": wandb.Image(CBCT[0], caption=str(path_CBCT)),
                "CT": wandb.Image(CT[0], caption=str(path_CT))
            }

            wandb.log(logdict)

        # Convert array to torch tensors
        CBCT = torch.tensor(CBCT)
        CT = torch.tensor(CT)

        CBCT, CT = self.slice_sampler.get_patch_pair(CBCT, CT)

        # Limits the lowest and highest HU unit
        CBCT = torch.clamp(CBCT, self.hu_min, self.hu_max)
        CT = torch.clamp(CT, self.hu_min, self.hu_max)

        # Normalize Hounsfield units to range [-1,1]
        CBCT = min_max_normalize(CBCT, self.hu_min, self.hu_max)
        CT = min_max_normalize(CT, self.hu_min, self.hu_max)

        # Add channel dimension
        CT = CT.unsqueeze(0)
        CBCT = CBCT.unsqueeze(0)

        return {'A': CBCT, 'B': CT}

    def __len__(self):
        return max(self.num_datapoints_CBCT, self.num_datapoints_CT)
