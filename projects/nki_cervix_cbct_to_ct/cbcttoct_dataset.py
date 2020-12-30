from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset

import midaGAN
from midaGAN.utils.io import make_recursive_dataset_of_files, load_json
from midaGAN.utils import sitk_utils
from midaGAN.data.utils.normalization import min_max_normalize, min_max_denormalize
from midaGAN.data.utils.registration_methods import register_CT_to_CBCT, truncate_CT_to_scope_of_CBCT
from midaGAN.data.utils.fov_truncate import truncate_CBCT_based_on_fov
from midaGAN.data.utils.body_mask import apply_body_mask_and_bound, get_body_mask_and_bound
from midaGAN.data.utils import pad
from midaGAN.data.utils.stochastic_focal_patching import StochasticFocalPatchSampler

# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN import configs

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
            CBCT = apply_body_mask_and_bound(CBCT, \
                    apply_mask=self.apply_mask, HU_threshold=self.cbct_mask_threshold)
        except:
            logger.error(f"Error applying mask and bound in file : {path_CBCT}")

        try:
            CT = apply_body_mask_and_bound(CT, \
                    apply_mask=self.apply_mask, HU_threshold=self.ct_mask_threshold)

        except:
            logger.error(f"Error applying mask and bound in file : {path_CT}")

        CBCT = pad(self.patch_size, CBCT)
        CT = pad(self.patch_size, CT)

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


# --------------------------- INFERENCE DATASET ----------------------------------------------
# --------------------------------------------------------------------------------------------


@dataclass
class CBCTtoCTInferenceDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "CBCTtoCTInferenceDataset"
    hounsfield_units_range: Tuple[int, int] = field(
        default_factory=lambda: (-1024, 2048))  #TODO: what should be the default range
    enable_masking: bool = False
    enable_bounding: bool = True
    cbct_mask_threshold: int = -700


class CBCTtoCTInferenceDataset(Dataset):

    def __init__(self, conf):
        # self.paths = make_dataset_of_directories(conf.dataset.root, EXTENSIONS)
        self.root_path = Path(conf.dataset.root).resolve()

        self.paths = []

        for patient in self.root_path.iterdir():
            self.paths.extend(make_recursive_dataset_of_files(patient / "CBCT", EXTENSIONS))

        self.num_datapoints = len(self.paths)
        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.dataset.hounsfield_units_range

        self.apply_mask = conf.dataset.enable_masking
        self.apply_bound = conf.dataset.enable_bounding
        self.cbct_mask_threshold = conf.dataset.cbct_mask_threshold

    def __getitem__(self, index):
        path = str(self.paths[index])

        print(path)
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

        if self.apply_mask or self.apply_bound:
            body_mask, ((z_max, z_min), \
            (y_max, y_min), (x_max, x_min)) = get_body_mask_and_bound(volume, self.cbct_mask_threshold)

            # Apply mask to the image array
            if self.apply_mask:
                volume = np.where(body_mask, volume, -1024)
                metadata.update({'mask': body_mask})

            # Index the array within the bounds and return cropped array
            if self.apply_bound:
                volume = volume[z_max:z_min, y_max:y_min, x_max:x_min]
                metadata.update({'bounds': ((z_max, z_min), (y_max, y_min), (x_max, x_min))})

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

    def save(self, tensor, metadata, output_dir):
        tensor = tensor.squeeze().cpu()
        tensor = min_max_denormalize(tensor, self.hu_min, self.hu_max)

        masking_value = torch.tensor(-1024, dtype=torch.float)

        if 'mask' in metadata or 'bounds' in metadata:

            full_tensor = torch.full(
                (metadata['size'][2], metadata['size'][1], metadata['size'][0]),
                masking_value,
                dtype=torch.float)

            if 'bounds' in metadata:
                bounds = metadata['bounds']
                full_tensor[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1],
                            bounds[2][0]:bounds[2][1]] = tensor

            if 'mask' in metadata:
                mask = torch.tensor(metadata['mask'], dtype=bool).squeeze()
                full_tensor = torch.where(mask, full_tensor, masking_value)

        else:
            full_tensor = tensor

        sitk_image = sitk_utils.tensor_to_sitk_image(full_tensor, metadata['origin'],
                                                     metadata['spacing'], metadata['direction'],
                                                     metadata['dtype'])

        # Dataset used has a directory per each datapoint, the name of each datapoint's dir is used to save the output
        datapoint_path = Path(str(metadata['path']))

        save_path = datapoint_path.relative_to(self.root_path)

        save_path = Path(output_dir) / save_path

        save_path.parent.mkdir(exist_ok=True, parents=True)

        sitk_utils.write(sitk_image, save_path)


# --------------------------- EVALUATION DATASET ---------------------------------------------
# --------------------------------------------------------------------------------------------


@dataclass
class CBCTtoCTEvalDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "CBCTtoCTEvalDataset"
    hounsfield_units_range: Tuple[int, int] = field(
        default_factory=lambda: (-1024, 2048))  #TODO: what should be the default range
    enable_masking: bool = True
    enable_bounding: bool = False
    cbct_mask_threshold: int = -700
    ct_mask_threshold: int = -300


class CBCTtoCTEvalDataset(Dataset):

    def __init__(self, conf):
        # self.paths = make_dataset_of_directories(conf.dataset.root, EXTENSIONS)
        self.root_path = Path(conf.dataset.root).resolve()

        self.paths = {}

        for patient in self.root_path.iterdir():

            # Sorted list of files is returned, pick the first CBCT volume.

            if (patient / "registered").is_dir():

                first_CBCT = (patient / "registered" / "target").with_suffix(".nrrd")
                dpCT = (patient / "registered" / "deformed").with_suffix(".nrrd")

                self.paths[patient] = {"CT": dpCT, "CBCT": first_CBCT}

        self.num_datapoints = len(self.paths)
        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.dataset.hounsfield_units_range

        self.apply_mask = conf.dataset.enable_masking
        self.apply_bound = conf.dataset.enable_bounding
        self.cbct_mask_threshold = conf.dataset.cbct_mask_threshold
        self.ct_mask_threshold = conf.dataset.ct_mask_threshold

    def __getitem__(self, index):
        patient_index = list(self.paths)[index]

        patient_dict = self.paths[patient_index]

        planning_CT = patient_dict["CT"]
        first_CBCT = patient_dict["CBCT"]

        # load nrrd as SimpleITK objects
        CT = sitk_utils.load(planning_CT)
        CBCT = sitk_utils.load(first_CBCT)

        metadata = {
            'path': str(first_CBCT),
            'size': CBCT.GetSize(),
            'origin': CBCT.GetOrigin(),
            'spacing': CBCT.GetSpacing(),
            'direction': CBCT.GetDirection(),
            'dtype': sitk_utils.get_npy_dtype(CBCT)
        }

        CBCT = sitk_utils.get_npy(CBCT)
        CT = sitk_utils.get_npy(CT)

        CT = torch.tensor(CT)
        CBCT = torch.tensor(CBCT)

        # Limits the lowest and highest HU unit
        CT = torch.clamp(CT, self.hu_min, self.hu_max)
        CBCT = torch.clamp(CBCT, self.hu_min, self.hu_max)
        # Normalize Hounsfield units to range [-1,1]
        CT = min_max_normalize(CT, self.hu_min, self.hu_max)
        CBCT = min_max_normalize(CBCT, self.hu_min, self.hu_max)
        # Add channel dimension (1 = grayscale)
        CT = CT.unsqueeze(0)
        CBCT = CBCT.unsqueeze(0)

        return {"A": CBCT, "B": CT, "metadata": metadata}

    def scale_to_HU(self, tensor):
        return min_max_denormalize(tensor.clone(), self.hu_min, self.hu_max)

    def save(self, tensor, metadata, output_dir):
        tensor = tensor.squeeze().cpu()
        tensor = min_max_denormalize(tensor, self.hu_min, self.hu_max)

        masking_value = torch.tensor(-1024, dtype=torch.float)

        if 'mask' in metadata or 'bounds' in metadata:

            full_tensor = torch.full(
                (metadata['size'][2], metadata['size'][1], metadata['size'][0]),
                masking_value,
                dtype=torch.float)

            if 'bounds' in metadata:
                bounds = metadata['bounds']
                full_tensor[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1],
                            bounds[2][0]:bounds[2][1]] = tensor

            if 'mask' in metadata:
                mask = torch.tensor(metadata['mask'], dtype=bool).squeeze()
                full_tensor = torch.where(mask, full_tensor, masking_value)

        else:
            full_tensor = tensor

        sitk_image = sitk_utils.tensor_to_sitk_image(full_tensor, metadata['origin'],
                                                     metadata['spacing'], metadata['direction'],
                                                     metadata['dtype'])

        # Dataset used has a directory per each datapoint, the name of each datapoint's dir is used to save the output
        datapoint_path = Path(str(metadata['path']))

        save_path = datapoint_path.relative_to(self.root_path)

        save_path = Path(output_dir) / save_path

        save_path.parent.mkdir(exist_ok=True, parents=True)

        sitk_utils.write(sitk_image, save_path)

    def __len__(self):
        return self.num_datapoints
