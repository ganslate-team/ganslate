from pathlib import Path
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

import midaGAN
from midaGAN.utils.io import make_dataset_of_directories, load_json
from midaGAN.utils import sitk_utils
from midaGAN.data.utils.normalization import min_max_normalize, min_max_denormalize
from midaGAN.data.utils.register_truncate import truncate_CT_to_scope_of_CBCT
from midaGAN.data.utils.stochastic_focal_patching import StochasticFocalPatchSampler

# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf import BaseDatasetConfig

logger = logging.getLogger(__name__)

EXTENSIONS = ['.nrrd']


@dataclass
class CBCTtoCTDatasetConfig(BaseDatasetConfig):
    name:                    str = "CBCTtoCTDataset"
    patch_size:              Tuple[int, int, int] = field(default_factory=lambda: (32, 32, 32))
    hounsfield_units_range:  Tuple[int, int] = field(default_factory=lambda: (-1000, 2000)) #TODO: what should be the default range
    focal_region_proportion: float = 0.2    # Proportion of focal region size compared to original volume size


class CBCTtoCTDataset(Dataset):
    def __init__(self, conf):
        dir_CBCT = Path(conf.dataset.root) / 'CBCT'
        dir_CT = Path(conf.dataset.root) / 'CT'
        self.paths_CBCT = make_dataset_of_directories(dir_CBCT, EXTENSIONS)
        self.paths_CT = make_dataset_of_directories(dir_CT, EXTENSIONS)
        self.num_datapoints_CBCT = len(self.paths_CBCT)
        self.num_datapoints_CT = len(self.paths_CT)

        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.dataset.hounsfield_units_range

        focal_region_proportion = conf.dataset.focal_region_proportion
        self.patch_size = np.array(conf.dataset.patch_size)
        self.patch_sampler = StochasticFocalPatchSampler(self.patch_size, focal_region_proportion)

    def __getitem__(self, index):
        index_CBCT = index % self.num_datapoints_CBCT
        index_CT = random.randint(0, self.num_datapoints_CT - 1)

        path_CBCT = Path(self.paths_CBCT[index_CBCT]) / 'CT.nrrd'
        path_CT = Path(self.paths_CT[index_CT]) / 'CT.nrrd'
        
        # load nrrd as SimpleITK objects
        CBCT = sitk_utils.load(path_CBCT)
        CT = sitk_utils.load(path_CT)

        # TODO: make a function
        if (sitk_utils.is_volume_smaller_than(CBCT, self.patch_size) 
                or sitk_utils.is_volume_smaller_than(CT, self.patch_size)):
            raise ValueError("Volume size not smaller than the defined patch size.\
                              \nCBCT: {} \nCT: {} \npatch_size: {}."\
                             .format(sitk_utils.get_size_zxy(CBCT),
                                     sitk_utils.get_size_zxy(CT), 
                                     self.patch_size))

	    # limit CT so that it only contains part of the body shown in CBCT
        CT_truncated = truncate_CT_to_scope_of_CBCT(CT, CBCT)
        if sitk_utils.is_volume_smaller_than(CT_truncated, self.patch_size):
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


@dataclass
class CBCTtoCTInferenceDatasetConfig(BaseDatasetConfig):
    name:                    str = "CBCTtoCTInferenceDataset"
    hounsfield_units_range:  Tuple[int, int] = field(default_factory=lambda: (-1000, 2000)) #TODO: what should be the default range
    

class CBCTtoCTInferenceDataset(Dataset):
    def __init__(self, conf):
        self.paths = make_dataset_of_directories(conf.dataset.root, EXTENSIONS)
        self.num_datapoints = len(self.paths)
        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.dataset.hounsfield_units_range

    def __getitem__(self, index):
        path = str(Path(self.paths[index]) / 'CT.nrrd')
        # load nrrd as SimpleITK objects
        volume = sitk_utils.load(path)
        metadata = (path, 
                    volume.GetOrigin(), 
                    volume.GetSpacing(), 
                    volume.GetDirection(),
                    sitk_utils.get_npy_dtype(volume))

        volume = sitk_utils.get_tensor(volume)
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
        tensor = tensor.squeeze()
        tensor = min_max_denormalize(tensor, self.hu_min, self.hu_max)
        
        datapoint_path, origin, spacing, direction, dtype = metadata
        sitk_image = sitk_utils.tensor_to_sitk_image(tensor, origin, spacing, direction, dtype)

        # Dataset used has a directory per each datapoint, the name of each datapoint's dir is used to save the output
        datapoint_name = Path(str(datapoint_path)).parent.name
        save_path = Path(output_dir) / Path(datapoint_name).with_suffix('.nrrd')

        sitk_utils.write(sitk_image, save_path)
        



