from pathlib import Path
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

import midaGAN
from midaGAN.utils import io, sitk_utils
from midaGAN.data.utils.ops import pad
from midaGAN.data.utils.normalization import min_max_normalize, min_max_denormalize
from midaGAN.data.utils.registration_methods import truncate_CT_to_scope_of_CBCT
from midaGAN.data.utils.fov_truncate import truncate_CBCT_based_on_fov
from midaGAN.data.utils.body_mask import apply_body_mask_and_bound
from midaGAN.data.utils.stochastic_focal_patching import StochasticFocalPatchSampler

# Config imports
from typing import Tuple
from dataclasses import dataclass
from omegaconf import MISSING
from midaGAN import configs

logger = logging.getLogger(__name__)

EXTENSIONS = ['.nrrd']


@dataclass
class CBCTtoCTDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "CBCTtoCTDataset"
    patch_size: Tuple[int, int, int] = (32, 32, 32)
    hounsfield_units_range: Tuple[int, int] = (-1000, 2000)
    # Proportion of focal region size compared to original volume size
    focal_region_proportion: float = 0.2


class CBCTtoCTDataset(Dataset):

    def __init__(self, conf):
        root_path = Path(conf.train.dataset.root).resolve()

        self.paths_CBCT = []
        self.paths_CT = []

        for patient in root_path.iterdir():

            cbct_dir = patient / "CBCT"
            if cbct_dir.is_dir():
                patient_cbcts = io.make_dataset_of_directories(cbct_dir, EXTENSIONS)
                self.paths_CBCT.extend(patient_cbcts)

            ct_dir = patient / "CT"
            if ct_dir.is_dir():
                patient_cts = io.make_dataset_of_directories(ct_dir, EXTENSIONS)
                self.paths_CT.extend(patient_cts)

        self.num_datapoints_CBCT = len(self.paths_CBCT)
        self.num_datapoints_CT = len(self.paths_CT)

        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.train.dataset.hounsfield_units_range

        focal_region_proportion = conf.train.dataset.focal_region_proportion
        self.patch_size = conf.train.dataset.patch_size
        self.patch_sampler = StochasticFocalPatchSampler(self.patch_size, focal_region_proportion)

    def __getitem__(self, index):

        index_CBCT = index % self.num_datapoints_CBCT
        index_CT = random.randint(0, self.num_datapoints_CT - 1)

        path_CBCT = self.paths_CBCT[index_CBCT]
        path_CT = self.paths_CT[index_CT]

        num_replacements_threshold = 10

        for i in reversed(range(num_replacements_threshold)):
            # load nrrd as SimpleITK objects
            CT = sitk_utils.load(Path(path_CT) / 'CT.nrrd')

            # Selected if it's big enough for the training
            if not sitk_utils.is_image_smaller_than(CT, self.patch_size):
                break

            # Remove from the paths since it's too small for the training.
            # If-statement since the path might already be deleted by another worker.
            if path_CT in self.paths_CT:
                self.paths_CT.remove(path_CT)
                logger.info(f"Removed the image with size {sitk_utils.get_torch_like_size(CT)}"
                            f" as it's smaller than the defined patch_size. Path: {path_CT}")

            # Randomly select another one
            path_CT = random.choice(self.paths_CT)

            if i == 0:
                raise ValueError(
                    f"Could not replace the image for {num_replacements_threshold}"
                    " consecutive times. Please verify your images and the specified config.")

        for i in reversed(range(num_replacements_threshold)):
            # load nrrd as SimpleITK objects
            CBCT = sitk_utils.load(Path(path_CBCT) / 'CBCT.nrrd')

            # Remove warped slices in CBCT
            num_slices = sitk_utils.get_size(CBCT)[-1]
            start_slice = int(num_slices * 0.13)
            end_slice = int(num_slices * 0.82)
            CBCT = sitk_utils.slice_image(CBCT, start=(0, 0, start_slice), end=(-1, -1, end_slice))

            # Selected if it's big enough for the training
            if not sitk_utils.is_image_smaller_than(CBCT, self.patch_size):
                break

            # Remove from the paths since it's too small for the training.
            # If-statement since the path might already be deleted by another worker.
            if path_CBCT in self.paths_CBCT:
                self.paths_CBCT.remove(path_CBCT)
                logger.info(f"Removed the image with size {sitk_utils.get_torch_like_size(CBCT)}"
                            f" as it's smaller than the defined patch_size. Path: {path_CBCT}")

            # Randomly select another one
            path_CBCT = random.choice(self.paths_CBCT)

            if i == 0:
                raise ValueError(
                    f"Could not replace the image for {num_replacements_threshold}"
                    " consecutive times. Please verify your images and the specified config.")

        CT = mask_out_ct(CT, path_CT, self.hu_min)

        # Limit CT so that it only contains part of the body shown in CBCT
        CT_truncated = truncate_CT_to_scope_of_CBCT(CT, CBCT)
        if sitk_utils.is_image_smaller_than(CT_truncated, self.patch_size):
            logger.info("Post-registration truncated CT is smaller than the defined patch size."
                        " Passing the whole CT volume.")
            del CT_truncated
        else:
            CT = CT_truncated

        # Mask and bound is applied on numpy arrays!
        CBCT = sitk_utils.get_npy(CBCT)
        CT = sitk_utils.get_npy(CT)

        CBCT = apply_body_mask_and_bound(CBCT,
                                         apply_mask=True,
                                         masking_value=self.hu_min,
                                         apply_bound=True,
                                         hu_threshold=-800)
        CBCT = pad(CBCT, self.patch_size)

        CT = apply_body_mask_and_bound(CT,
                                       apply_mask=True,
                                       masking_value=self.hu_min,
                                       apply_bound=True,
                                       hu_threshold=-600)
        CT = pad(CT, self.patch_size)

        CBCT = torch.tensor(CBCT)
        CT = torch.tensor(CT)

        # Extract patches
        CBCT, CT = self.patch_sampler.get_patch_pair(CBCT, CT)

        CBCT, CT = clamp_normalize(CBCT, CT, self.hu_min, self.hu_max)

        return {'A': CBCT, 'B': CT}

    def __len__(self):
        return max(self.num_datapoints_CBCT, self.num_datapoints_CT)


def mask_out_ct(ct_scan, ct_dir_path, masking_value):
    # Use, by priority, BODY, External or treatment table mask
    # to mask out unuseful values in the CT
    mask_exists = True
    negated_mask = False

    if (ct_dir_path / 'BODY.nrrd').exists():
        mask_path = ct_dir_path / 'BODY.nrrd'
    elif paths := io.find_paths_containing_pattern(ct_dir_path, "External*"):
        mask_path = paths[0]
    elif paths := io.find_paths_containing_pattern(ct_dir_path, "*_treatment_*"):
        mask_path = paths[0]
        # Use negated masking as we want to only mask out the table
        negated_mask = True
    else:
        mask_exists = False
        logger.info(f'No mask files found for {ct_dir_path}')

    if mask_exists:
        ct_mask = sitk_utils.load(mask_path)
        ct_scan = sitk_utils.apply_mask(ct_scan,
                                        ct_mask,
                                        masking_value=masking_value,
                                        set_same_origin=True,
                                        negated_mask=negated_mask)
    return ct_scan


def clamp_normalize(A, B, min_value, max_value):
    # Limits the lowest and highest HU unit
    A = torch.clamp(A, min_value, max_value)
    B = torch.clamp(B, min_value, max_value)

    # Normalize Hounsfield units to range [-1,1]
    A = min_max_normalize(A, min_value, max_value)
    B = min_max_normalize(B, min_value, max_value)

    # Add channel dimension (1 = grayscale)
    A = A.unsqueeze(0)
    B = B.unsqueeze(0)

    return A, B
