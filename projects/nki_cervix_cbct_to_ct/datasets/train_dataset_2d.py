from pathlib import Path
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

import midaGAN
from midaGAN.utils.io import make_recursive_dataset_of_files, load_json
from midaGAN.utils import sitk_utils
from midaGAN.data.utils.normalization import min_max_normalize, min_max_denormalize
from midaGAN.data.utils.registration_methods import truncate_CT_to_scope_of_CBCT
from midaGAN.data.utils.fov_truncate import truncate_CBCT_based_on_fov
from midaGAN.data.utils.body_mask import apply_body_mask
from midaGAN.data.utils.stochastic_focal_patching import StochasticFocalPatchSampler

# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN import configs

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
        self.hu_min, self.hu_max = conf.train.dataset.hounsfield_units_range

        focal_region_proportion = conf.train.dataset.focal_region_proportion
        self.patch_size = np.array(conf.train.dataset.image_size)
        self.slice_sampler = StochasticFocalPatchSampler(self.patch_size, focal_region_proportion)

        self.apply_mask = conf.train.dataset.enable_masking
        self.cbct_mask_threshold = conf.train.dataset.cbct_mask_threshold
        self.ct_mask_threshold = conf.train.dataset.ct_mask_threshold

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

        CT_truncated = truncate_CT_to_scope_of_CBCT(CT, CBCT)
        if sitk_utils.is_image_smaller_than(CT_truncated, self.patch_size):
            logger.info(
                "Post-registration truncated CT is smaller than the defined patch size. Passing the whole CT volume."
            )
            del CT_truncated
        else:
            CT = CT_truncated

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

        if DEBUG:
            import wandb

            logdict = {
                "CBCT_3D": wandb.Image(CBCT[CBCT.shape[0] // 2], caption=str(path_CBCT)),
                "CT_3D": wandb.Image(CT[CT.shape[0] // 2], caption=str(path_CT))
            }
            wandb.log(logdict)

        # Convert array to torch tensors
        CBCT = torch.tensor(CBCT)
        CT = torch.tensor(CT)

        CBCT, CT = self.slice_sampler.get_patch_pair(CBCT, CT)

        if DEBUG:
            import wandb

            logdict = {
                "CBCT": wandb.Image(CBCT[0], caption=str(path_CBCT)),
                "CT": wandb.Image(CT[0], caption=str(path_CT))
            }

            wandb.log(logdict)

        CBCT = torch.tensor(CBCT)
        CT = torch.tensor(CT)

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
        return self.num_datapoints
