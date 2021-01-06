from pathlib import Path
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

from midaGAN.utils.io import make_dataset_of_files
from midaGAN.utils import sitk_utils
from midaGAN.data.utils.normalization import min_max_normalize, z_score_normalize
from midaGAN.data.utils.stochastic_focal_patching import StochasticFocalPatchSampler

# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN import configs

EXTENSIONS = ['.nii']

@dataclass
class RIREDatasetConfig(configs.base.BaseDatasetConfig):
    name:         str = "BratsDataset"
    patch_size:   Tuple[int, int, int] = (32, 32, 32)
    focal_region_proportion: float = 0    # Proportion of focal region size compared to original volume size
    hounsfield_units_range:  Tuple[int, int] = field(default_factory=lambda: (-1000, 2000)) #TODO: what should be the default range

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
        CT = torch.clamp(CT, self.hu_min, self.hu_max)

        # # Normalize Hounsfield units to range [-1,1] for CT
        # CT = min_max_normalize(CT, self.hu_min, self.hu_max)

        # Z-score standardization per volume
        MR = z_score_normalize(MR, scale_to_range=(-1,1))
        CT = z_score_normalize(CT, scale_to_range=(-1,1))
        
        # Add channel dimension (1 = grayscale)
        MR = MR.unsqueeze(0)
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
        # Z-score standardization per volume
        volume = z_score_normalize(volume, scale_to_range=(-1,1))
        # Add channel dimension (1 = grayscale)
        volume = volume.unsqueeze(0)

        return volume, metadata

    def __len__(self):
        return self.num_datapoints


    def save(self, tensor, metadata, output_dir):
        tensor = tensor.squeeze()
        # tensor = min_max_denormalize(tensor, self.hu_min, self.hu_max)
        # TODO: Fix fake denormalization
        tensor = 341*tensor+330

        datapoint_path, origin, spacing, direction, dtype = metadata
        sitk_image = sitk_utils.tensor_to_sitk_image(tensor, origin, spacing, direction, dtype)

        # Name of each datapoint is used to save the output
        datapoint_name = Path(str(datapoint_path)).name
        save_path = Path(output_dir) / Path(datapoint_name).with_suffix('.nii')

        sitk_utils.write(sitk_image, save_path)