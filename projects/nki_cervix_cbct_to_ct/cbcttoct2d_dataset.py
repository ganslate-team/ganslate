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
from midaGAN.data.utils.register_truncate import truncate_CT_to_scope_of_CBCT
from midaGAN.data.utils.fov_truncate import truncate_CBCT_based_on_fov
from midaGAN.data.utils.body_mask import apply_body_mask_and_bound, get_body_mask_and_bound
from midaGAN.data.utils import size_invalid_check_and_replace
from midaGAN.data.utils.slice_sampler import SliceSampler

# Config imports
from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf import BaseDatasetConfig

logger = logging.getLogger(__name__)

EXTENSIONS = ['.nrrd']
DEBUG = False

@dataclass
class CBCTtoCT2DDatasetConfig(BaseDatasetConfig):
    name:                    str = "CBCTtoCTDataset"
    load_size:               int = 256
    hounsfield_units_range:  Tuple[int, int] = field(default_factory=lambda: (-1000, 2000)) #TODO: what should be the default range
    select_origin_for_axes: Tuple[bool] = (False, True, True)
    focal_region_proportion: float = 0.2    # Proportion of focal region size compared to original volume size
    enable_cache:            bool = False
    image_channels:          int = 1
    enable_masking:          bool = True
    enable_bounding:         bool = True
    ct_mask_threshold:          int = -300
    cbct_mask_threshold:        int = -700


class CBCTtoCT2DDataset(Dataset):
    def __init__(self, conf):

        root_path = Path(conf.dataset.root).resolve()
        
        self.paths_CBCT = {}
        self.paths_CT = {}

        for patient in root_path.iterdir():
            self.paths_CBCT[patient.stem] = make_recursive_dataset_of_files(patient / "CBCT", EXTENSIONS)
            CT_nrrds = make_recursive_dataset_of_files(patient / "CT", EXTENSIONS)
            self.paths_CT[patient.stem] = [path for path in CT_nrrds if path.stem == "CT"]

        assert len(self.paths_CBCT) == len(self.paths_CT), \
            "Number of patients should match for CBCT and CT"

        self.num_datapoints = len(self.paths_CT)

        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.dataset.hounsfield_units_range

        focal_region_proportion = conf.dataset.focal_region_proportion
        self.patch_size = np.array([conf.dataset.load_size]*2)
        self.slice_sampler = SliceSampler(self.patch_size, focal_region_proportion)

        self.apply_mask = conf.dataset.enable_masking
        self.apply_bound = conf.dataset.enable_bounding
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


        CT_truncated = truncate_CT_to_scope_of_CBCT(CT, CBCT)
        if sitk_utils.is_image_smaller_than(CT_truncated, self.patch_size):
            logger.info("Post-registration truncated CT is smaller than the defined patch size. Passing the whole CT volume.")
            del CT_truncated
        else:
            CT = CT_truncated



        # Mask and bound is applied on numpy arrays!
        CBCT = sitk_utils.get_npy(CBCT)
        CT = sitk_utils.get_npy(CT)

        # Apply body masking to the CT and CBCT arrays 
        # and bound the z, x, y grid to around the mask
        try: 
            CBCT = apply_body_mask_and_bound(CBCT, \
                    apply_mask=self.apply_mask, apply_bound=self.apply_bound, HU_threshold=self.cbct_mask_threshold)
        except:
            logger.error(f"Error applying mask and bound in file : {path_CBCT}")

        try:
            CT = apply_body_mask_and_bound(CT, \
                    apply_mask=self.apply_mask, apply_bound=self.apply_bound, HU_threshold=self.ct_mask_threshold)

        except:
            logger.error(f"Error applying mask and bound in file : {path_CT}")        

        

        if DEBUG:
            import wandb

            logdict = {
            "CBCT": wandb.Image(CBCT[CBCT.shape[0]//2], caption=str(path_CBCT)),
            "CT":wandb.Image(CT[CT.shape[0]//2], caption=str(path_CT))
            }

            wandb.log(logdict)

        # Convert array to torch tensors
        CBCT = torch.tensor(CBCT)
        CT = torch.tensor(CT)

        CBCT_slice, CT_slice = self.slice_sampler.get_slice_pair(CBCT, CT) 

        CBCT_slice = torch.tensor(CBCT_slice)
        CT_slice = torch.tensor(CT_slice)

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
        return self.num_datapoints


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
        



