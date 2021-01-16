from pathlib import Path
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

from midaGAN.utils.io import make_dataset_of_files
from midaGAN.utils import sitk_utils
from midaGAN.data.utils.normalization import min_max_normalize, min_max_denormalize, z_score_normalize, unequal_normalize, unequal_denormalize
from midaGAN.data.utils.stochastic_focal_patching import StochasticFocalPatchSampler

# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN import configs

EXTENSIONS = ['.nii']

@dataclass
class RIREDatasetConfig(configs.base.BaseDatasetConfig):
    name:         str = "RIREDataset"
    patch_size:   Tuple[int, int, int] = (16, 128, 128)
    focal_region_proportion: float = 0    # Proportion of focal region size compared to original volume size
    hounsfield_units_range:  Tuple[int, int] = field(default_factory=lambda: (-1000, 2000))

def get_sequence_or_ct(sitk_image):
    size = list(sitk_image.GetSize())
    index = [0, 0, 0]

    Extractor = sitk.ExtractImageFilter()
    Extractor.SetSize(size)
    Extractor.SetIndex(index)
    return Extractor.Execute(sitk_image)


class RIREDataset(Dataset):
    def __init__(self, conf):
        dir_MR = Path(conf.dataset.root) / 'MR'
        dir_CT = Path(conf.dataset.root) / 'CT'
        self.unequal_normalize = conf.dataset.unequal_normalize
        self.paths_MR = make_dataset_of_files(dir_MR, EXTENSIONS)
        self.paths_CT = make_dataset_of_files(dir_CT, EXTENSIONS)
        self.num_datapoints_MR = len(self.paths_MR)
        self.num_datapoints_CT = len(self.paths_CT)

        focal_region_proportion = conf.dataset.focal_region_proportion
        self.patch_size = np.array(conf.dataset.patch_size)
        self.patch_sampler = StochasticFocalPatchSampler(self.patch_size, focal_region_proportion)

        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.dataset.hounsfield_units_range

    def __getitem__(self, index):
        index_MR = index % self.num_datapoints_MR
        index_CT = random.randint(0, self.num_datapoints_CT - 1)

        path_MR = self.paths_MR[index_MR]
        path_CT = self.paths_CT[index_CT]
        
        # load nrrd as SimpleITK objects
        MR = sitk_utils.load(path_MR)
        CT = sitk_utils.load(path_CT)

        MR = get_sequence_or_ct(MR)
        CT = get_sequence_or_ct(CT)

        if (sitk_utils.is_image_smaller_than(MR, self.patch_size) 
                or sitk_utils.is_image_smaller_than(CT, self.patch_size)):
            raise ValueError("Volume size not smaller than the defined patch size.\
                              \nA: {} \nB: {} \npatch_size: {}."\
                             .format(sitk_utils.get_size_zxy(MR),
                                     sitk_utils.get_size_zxy(CT), 
                                     self.patch_size))

        MR = sitk_utils.get_tensor(MR)
        CT = sitk_utils.get_tensor(CT)

        # no limitation of MR or CT needed because volumes were pre-processed to be coregistered

        # Extract patches
        MR, CT = self.patch_sampler.get_patch_pair(MR, CT)

        # Limits the lowest and highest HU unit for the CT
        # Note: RIRE data was originally given with -1024 as minimal HU value
        # However there should be practically no information loss between -1024 and -1000
        CT = torch.clamp(CT, self.hu_min, self.hu_max)

        # Normalize MR to range [-1,1]
        # Only meaningful due to preprocessing of MR via histogram matching
        min_MR = 0
        max_MR = 1695
        MR = min_max_normalize(MR, min_MR, max_MR)

        min_value = -1000
        max_value = 2000
        if self.unequal_normalize:
            # unequal_normalize normalizes CT to range [-1,1] but according to piecewise linear function
            # This is an unequal normalization that accentuates the expressivity of the range [-50, 150] (important for brain anatomy)
            split_points = [-50, 150]
            split_proportions = [0.25, 0.5, 0.25]
            CT = unequal_normalize(CT.detach().numpy(), min_value, max_value, split_points, split_proportions)
        else:
            CT = min_max_normalize(CT, min_value, max_value)
        
        # Add channel dimension (1 = grayscale)
        MR = MR.unsqueeze(0)
        if self.unequal_normalize:
            CT = torch.from_numpy(CT).unsqueeze(0)
        else:
            CT = CT.unsqueeze(0)

        return {'A': MR, 'B': CT}

    def __len__(self):
        return max(self.num_datapoints_MR, self.num_datapoints_CT)


@dataclass
class RIREInferenceDatasetConfig(configs.base.BaseDatasetConfig):
    name:                    str = "RIREInferenceDataset"
    hounsfield_units_range:  Tuple[int, int] = field(default_factory=lambda: (-1000, 2000)) #TODO: what should be the default range
    

class RIREInferenceDataset(Dataset):
    def __init__(self, conf):
        self.paths = make_dataset_of_files(conf.dataset.root, EXTENSIONS)
        self.num_datapoints = len(self.paths)
        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.dataset.hounsfield_units_range

    def __getitem__(self, index):
        path = str(Path(self.paths[index]))
        # load nrrd as SimpleITK objects
        volume = sitk_utils.load(path)
        volume = get_sequence_or_ct(volume)
        metadata = (path, 
                    volume.GetOrigin(), 
                    volume.GetSpacing(), 
                    volume.GetDirection(),
                    sitk_utils.get_npy_dtype(volume))

        volume = sitk_utils.get_tensor(volume)
        # Normalize MR to range [-1,1]
        # Only meaningful due to preprocessing of MR via histogram matching
        volume = min_max_normalize(volume, 0, 1695)
        # Add channel dimension (1 = grayscale)
        volume = volume.unsqueeze(0)

        return volume, metadata

    def __len__(self):
        return self.num_datapoints


    def save(self, tensor, metadata, output_dir):
        tensor = tensor.squeeze()

        # denormalize CT according to piecewise linear function
        # Resulting from an unequal normalization that accentuates the expressivity of the range [-50, 150] (important for brain anatomy)
        min_value = -1000
        max_value = 2000
        split_points = [-50, 150]
        split_proportions = [0.25, 0.5, 0.25]
        tensor = unequal_denormalize(tensor, min_value, max_value, split_points, split_proportions)

        datapoint_path, origin, spacing, direction, dtype = metadata
        sitk_image = sitk_utils.tensor_to_sitk_image(tensor, origin, spacing, direction, dtype)

        # Name of each datapoint is used to save the output
        datapoint_name = Path(str(datapoint_path)).name
        save_path = Path(output_dir) / Path(datapoint_name).with_suffix('.nii')

        sitk_utils.write(sitk_image, save_path)