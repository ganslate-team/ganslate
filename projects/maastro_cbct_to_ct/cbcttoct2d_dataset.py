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
from midaGAN.data.utils.slice_sampler import SliceSampler

# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf import BaseDatasetConfig

logger = logging.getLogger(__name__)

EXTENSIONS = ['.nrrd']


@dataclass
class CBCTtoCT2DDatasetConfig(BaseDatasetConfig):
    name:                    str = "CBCTtoCTDataset"
    load_size:               int = 256
    hounsfield_units_range:  Tuple[int, int] = field(default_factory=lambda: (-1000, 2000)) #TODO: what should be the default range
    focal_region_proportion: float = 0.2    # Proportion of focal region size compared to original volume size
    enable_cache:            bool = False
    image_channels:          int = 1

class CBCTtoCT2DDataset(Dataset):
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
        self.patch_size = np.array([conf.dataset.load_size]*2)
        self.slice_sampler = SliceSampler(self.patch_size, focal_region_proportion)
        self.conf = conf
        self.data_cache = {}


    def add_to_cache(self, path, data):
        self.data_cache[path] = data

    def clear_cache(self):
        self.data_cache = {}

    def __getitem__(self, index):
        index_CBCT = index % self.num_datapoints_CBCT
        index_CT = random.randint(0, self.num_datapoints_CT - 1)

        path_CBCT = Path(self.paths_CBCT[index_CBCT]) / 'CT.nrrd'
        path_CT = Path(self.paths_CT[index_CT]) / 'CT.nrrd'
        
        # load nrrd as SimpleITK objects

        if self.conf.dataset.enable_cache:
            if path_CBCT not in self.data_cache:
                CBCT = sitk_utils.load(path_CBCT)
                self.add_to_cache(path_CBCT, CBCT)
            else:
                CBCT = self.data_cache[path_CBCT]

            if path_CT not in self.data_cache:
                CT = sitk_utils.load(path_CT)
                self.add_to_cache(path_CT, CT)
            else:
                CT = self.data_cache[path_CT]
        else:
            CBCT = sitk_utils.load(path_CBCT)
            CT = sitk_utils.load(path_CT)

        CBCT = sitk_utils.get_npy(CBCT)
        CT = sitk_utils.get_npy(CT)

        # Extract patches
        CBCT_slice, CT_slice = self.slice_sampler.get_slice_pair(CBCT, CT) 

        CBCT_slice = torch.Tensor(CBCT_slice)
        CT_slice = torch.Tensor(CT_slice)

        # Limits the lowest and highest HU unit
        CBCT_slice = torch.clamp(CBCT_slice, self.hu_min, self.hu_max)
        CT_slice = torch.clamp(CT_slice, self.hu_min, self.hu_max)

        # Normalize Hounsfield units to range [-1,1]
        CBCT_slice = min_max_normalize(CBCT_slice, self.hu_min, self.hu_max)
        CT_slice = min_max_normalize(CT_slice, self.hu_min, self.hu_max)

        # Add channel dimension (1 = grayscale)
        CBCT_slice = CBCT_slice.unsqueeze(0)
        CT_slice = CT_slice.unsqueeze(0)

        return {'A': CBCT_slice, 'B': CT_slice}

    def __len__(self):
        return max(self.num_datapoints_CBCT, self.num_datapoints_CT)


# @dataclass
# class CBCTtoCT2DInferenceDatasetConfig(BaseDatasetConfig):
#     name:                    str = "CBCTtoCTInferenceDataset"
#     hounsfield_units_range:  Tuple[int, int] = field(default_factory=lambda: (-1000, 2000)) #TODO: what should be the default range
    

# class CBCTtoCT2DInferenceDataset(Dataset):
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
        



