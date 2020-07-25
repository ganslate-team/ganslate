import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.io import make_dataset_of_folders, load_json
from utils.medical_imaging import normalize_from_hu
from utils import sitk_utils
from data.utils.register_truncate import truncate_CT_to_scope_of_CBCT
from data.utils.stochastic_focal_patching import StochasticFocalPatchSampler

EXTENSIONS = ['.nrrd']

class CBCTtoCTDataset(Dataset):
    def __init__(self, conf):
        dir_CBCT = os.path.join(conf.dataset.root, 'CBCT')
        dir_CT = os.path.join(conf.dataset.root, 'CT')
        self.paths_CBCT = make_dataset_of_folders(dir_CBCT, EXTENSIONS)
        self.paths_CT = make_dataset_of_folders(dir_CT, EXTENSIONS)
        self.num_datapoints_CBCT = len(self.paths_CBCT)
        self.num_datapoints_CT = len(self.paths_CT)

        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.dataset.hounsfield_units_range

        focal_region_proportion = conf.dataset.focal_region_proportion
        self.patch_size = np.array(conf.dataset.patch_size)
        self.patch_sampler = StochasticFocalPatchSampler(self.patch_size, focal_region_proportion)

    def _is_volume_smaller_than_patch(self, sitk_volume):
        volume_size = sitk_utils.get_size_zxy(sitk_volume)
        if (volume_size < self.patch_size).any():
            return True
        return False

    def __getitem__(self, index):
        index_CBCT = index % self.num_datapoints_CBCT
        index_CT = random.randint(0, self.num_datapoints_CT - 1)

        path_CBCT = os.path.join(self.paths_CBCT[index_CBCT], 'CT.nrrd')
        path_CT = os.path.join(self.paths_CT[index_CT], 'CT.nrrd')
        
        # load nrrd as SimpleITK objects
        CBCT = sitk_utils.load(path_CBCT)
        CT = sitk_utils.load(path_CT)

        if self._is_volume_smaller_than_patch(CBCT) or self._is_volume_smaller_than_patch(CT):
            raise ValueError("Volume size not smaller than the defined patch size.\
                              \nCBCT: {} \nCT: {} \npatch_size: {}."\
                             .format(sitk_utils.get_size_zxy(CBCT),
                                     sitk_utils.get_size_zxy(CT), 
                                     self.patch_size))

	    # limit CT so that it only contains part of the body shown in CBCT
        CT_truncated = truncate_CT_to_scope_of_CBCT(CT, CBCT)
        if self._is_volume_smaller_than_patch(CT_truncated):
            print("Post-registration truncated CT is smaller than the defined patch size. Passing the whole CT volume.")
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
        CBCT = normalize_from_hu(CBCT, self.hu_min, self.hu_max)
        CT = normalize_from_hu(CT, self.hu_min, self.hu_max)

        # Add channel dimension (1 = grayscale)
        CBCT = CBCT.unsqueeze(0)
        CT = CT.unsqueeze(0)

        return {'A': CBCT, 'B': CT}


    def __len__(self):
        return max(self.num_datapoints_CBCT, self.num_datapoints_CT)




