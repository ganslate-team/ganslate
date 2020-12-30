from pathlib import Path
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

import midaGAN
from midaGAN.utils.io import make_dataset_of_directories
from midaGAN.utils import sitk_utils
from midaGAN.data.utils.normalization import min_max_normalize, min_max_denormalize
from midaGAN.data.utils.registration_methods import truncate_CT_to_scope_of_CBCT
from midaGAN.data.utils.fov_truncate import truncate_CBCT_based_on_fov

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
    name:                    str = "CBCTtoCTDataset"
    patch_size:              Tuple[int, int, int] = (32, 32, 32)
    hounsfield_units_range:  Tuple[int, int] = (-1000, 2000)
     # Proportion of focal region size compared to original volume size
    focal_region_proportion: float = 0.2


class CBCTtoCTDataset(Dataset):
    def __init__(self, conf):
        root_path = Path(conf.dataset.root).resolve()
        
        self.paths_CBCT = []
        self.paths_CT = []
        for patient in root_path.iterdir():
            if (patient / "CBCT").is_dir():
                self.paths_CBCT.extend(make_dataset_of_directories(patient / "CBCT", EXTENSIONS)) 
            if (patient / "CT").is_dir():
                self.paths_CT.extend(make_dataset_of_directories(patient / "CT", EXTENSIONS)) 

        self.num_datapoints_CBCT = len(self.paths_CBCT)
        self.num_datapoints_CT = len(self.paths_CT)

        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.dataset.hounsfield_units_range

        focal_region_proportion = conf.dataset.focal_region_proportion
        self.patch_size = conf.dataset.patch_size
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

            # Randomly select another one
            paths_CBCT = random.choice(self.paths_CBCT)
            
            if i == 0:
                raise ValueError(f"Could not replace the image for {num_replacements_threshold}"
                                 " consecutive times. Please verify your images and the specified config.")

        for i in reversed(range(num_replacements_threshold)):
            # load nrrd as SimpleITK objects
            CBCT = sitk_utils.load(Path(path_CBCT) / 'CBCT.nrrd')

            # Remove warped slices in CBCT
            num_slices = sitk_utils.get_size(CBCT)[-1]
            start_slice = int(num_slices * 0.13)
            end_slice = int(num_slices * 0.82)
            CBCT = sitk_utils.slice_image(CBCT,
                                          start=(0, 0, start_slice),
                                          end=(-1, -1, end_slice))

            # Selected if it's big enough for the training
            if not sitk_utils.is_image_smaller_than(CBCT, self.patch_size):
                break
            
            # Remove from the paths since it's too small for the training.
            # If-statement since the path might already be deleted by another worker. 
            if path_CBCT in self.paths_CBCT:
                self.paths_CBCT.remove(path_CBCT)

            # Randomly select another one
            paths_CBCT = random.choice(self.paths_CBCT)

            if i == 0:
                raise ValueError(f"Could not replace the image for {num_replacements_threshold}"
                                 " consecutive times. Please verify your images and the specified config.")

	    # limit CT so that it only contains part of the body shown in CBCT
        CT_truncated = truncate_CT_to_scope_of_CBCT(CT, CBCT)
        if sitk_utils.is_image_smaller_than(CT_truncated, self.patch_size):
            logger.info("Post-registration truncated CT is smaller than the defined patch size. Passing the whole CT volume.")
            del CT_truncated
        else:
            CT = CT_truncated

        CBCT = sitk_utils.get_tensor(CBCT)
        CT = sitk_utils.get_tensor(CT)

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
        return max(self.num_datapoints_CBCT, self.num_datapoints_CT)


# @dataclass
# class CBCTtoCTInferenceDatasetConfig(configs.base.BaseDatasetConfig):
#     name:                    str = "CBCTtoCTInferenceDataset"
#     hounsfield_units_range:  Tuple[int, int] = field(default_factory=lambda: (-1000, 2000)) #TODO: what should be the default range
    

# class CBCTtoCTInferenceDataset(Dataset):
#     def __init__(self, conf):
#         self.paths = make_dataset_of_directories(conf.dataset.root, EXTENSIONS)
#         self.num_datapoints = len(self.paths)
#         # Min and max HU values for clipping and normalization
#         self.hu_min, self.hu_max = conf.dataset.hounsfield_units_range

#     def __getitem__(self, index):
#         path = str(Path(self.paths[index]) / 'CT.nrrd')
#         # load nrrd as SimpleITK objects
#         volume = sitk_utils.load(path)
#         metadata = (path, 
#                     volume.GetOrigin(), 
#                     volume.GetSpacing(), 
#                     volume.GetDirection(),
#                     sitk_utils.get_npy_dtype(volume))

#         volume = sitk_utils.get_tensor(volume)
#         # Limits the lowest and highest HU unit
#         volume = torch.clamp(volume, self.hu_min, self.hu_max)
#         # Normalize Hounsfield units to range [-1,1]
#         volume = min_max_normalize(volume, self.hu_min, self.hu_max)
#         # Add channel dimension (1 = grayscale)
#         volume = volume.unsqueeze(0)

#         return volume, metadata

#     def __len__(self):
#         return self.num_datapoints

#     def save(self, tensor, metadata, output_dir):
#         tensor = tensor.squeeze()
#         tensor = min_max_denormalize(tensor, self.hu_min, self.hu_max)
        
#         datapoint_path, origin, spacing, direction, dtype = metadata
#         sitk_image = sitk_utils.tensor_to_sitk_image(tensor, origin, spacing, direction, dtype)

#         # Dataset used has a directory per each datapoint, the name of each datapoint's dir is used to save the output
#         datapoint_name = Path(str(datapoint_path)).parent.name
#         save_path = Path(output_dir) / Path(datapoint_name).with_suffix('.nrrd')

#         sitk_utils.write(sitk_image, save_path)
        



