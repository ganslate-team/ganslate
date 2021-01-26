import random
from dataclasses import dataclass, field
from pathlib import Path
# Config imports
from typing import Tuple

import midaGAN
import numpy as np
import torch
from midaGAN import configs
from midaGAN.data.utils import pad
from midaGAN.data.utils.body_mask import (apply_body_mask_and_bound,
                                          get_body_mask_and_bound)
from midaGAN.data.utils.fov_truncate import truncate_CBCT_based_on_fov
from midaGAN.data.utils.normalization import (min_max_denormalize,
                                              min_max_normalize)
from midaGAN.data.utils.registration_methods import (
    register_CT_to_CBCT, truncate_CT_to_scope_of_CBCT)
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

# --------------------------- TRAIN DATASET --------------------------------------------------
# --------------------------------------------------------------------------------------------


@dataclass
class CBCTtoCTDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "CBCTtoCTDataset"
    patch_size: Tuple[int, int, int] = field(default_factory=lambda: (32, 32, 32))
    hounsfield_units_range: Tuple[int, int] = field(
        default_factory=lambda: (-1024, 2048))  #TODO: what should be the default range
    focal_region_proportion: float = 0.2  # Proportion of focal region size compared to original volume size
    enable_masking: bool = True

    enable_bounding: bool = False  # Retained for legacy support. has no effect

    ct_mask_threshold: int = -300
    cbct_mask_threshold: int = -700


class CBCTtoCTDataset(Dataset):

    def __init__(self, conf):

        root_path = Path(conf.dataset.root).resolve()

        self.paths_CBCT = {}
        self.paths_CT = {}

        for patient in root_path.iterdir():
            self.paths_CBCT[patient.stem] = make_recursive_dataset_of_files(
                patient / "CBCT", EXTENSIONS)
            CT_nrrds = make_recursive_dataset_of_files(patient / "CT", EXTENSIONS)
            self.paths_CT[patient.stem] = [path for path in CT_nrrds if path.stem == "CT"]

        assert len(self.paths_CBCT) == len(self.paths_CT), \
            "Number of patients should match for CBCT and CT"

        self.num_datapoints = len(self.paths_CT)

        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.dataset.hounsfield_units_range

        focal_region_proportion = conf.dataset.focal_region_proportion
        self.patch_size = np.array(conf.dataset.patch_size)
        self.patch_sampler = StochasticFocalPatchSampler(self.patch_size, focal_region_proportion)

        self.apply_mask = conf.dataset.enable_masking
        self.cbct_mask_threshold = conf.dataset.cbct_mask_threshold
        self.ct_mask_threshold = conf.dataset.ct_mask_threshold

    def __getitem__(self, index):
        patient_index = list(self.paths_CT)[index]

        paths_CBCT = self.paths_CBCT[patient_index]
        paths_CT = self.paths_CT[patient_index]

        path_CBCT = random.choice(paths_CBCT)
        path_CT = random.choice(paths_CT)

        # load nrrd as SimpleITK objects
        CBCT = sitk_utils.load(path_CBCT)
        CT = sitk_utils.load(path_CT)

        # Subtract 1024 from CBCT to map values from grayscale to HU approx
        CBCT = CBCT - 1024

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
            CBCT = apply_body_mask_and_bound(CBCT,
                                             apply_mask=self.apply_mask,
                                             hu_threshold=self.cbct_mask_threshold)
        except:
            logger.error(f"Error applying mask and bound in file : {path_CBCT}")

        try:
            CT = apply_body_mask_and_bound(CT,
                                           apply_mask=self.apply_mask,
                                           hu_threshold=self.ct_mask_threshold)
        except:
            logger.error(f"Error applying mask and bound in file : {path_CT}")

        CBCT = pad(CBCT, self.patch_size)
        CT = pad(CT, self.patch_size)

        if DEBUG:
            import wandb

            logdict = {
                "CBCT": wandb.Image(CBCT[CBCT.shape[0] // 2], caption=str(path_CBCT)),
                "CT": wandb.Image(CT[CT.shape[0] // 2], caption=str(path_CT))
            }

            wandb.log(logdict)

        # Convert array to torch tensors
        CBCT = torch.tensor(CBCT)
        CT = torch.tensor(CT)

        # Extract patches
        CBCT, CT = self.patch_sampler.get_patch_pair(CBCT, CT)

        # Limits the lowest and highest HU unit
        CBCT = torch.clamp(CBCT, self.hu_min, self.hu_max)
        CT = torch.clamp(CT, self.hu_min, self.hu_max)

        # Normalize Hounsfield units to range [-1,1]
        CBCT = min_max_normalize(CBCT, self.hu_min, self.hu_max)
        CT = min_max_normalize(CT, self.hu_min, self.hu_max)

        # Add channel dimension (1 = grayscale)
        CBCT = CBCT.unsqueeze(0)
        CT = CT.unsqueeze(0)

        return {'A': CBCT, 'B': CT}

    def __len__(self):
        return self.num_datapoints
