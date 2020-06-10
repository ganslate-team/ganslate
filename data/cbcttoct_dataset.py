import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from util.file_utils import make_dataset_of_folders, load_json
from util.preprocessing import normalize_from_hu
from util import sitk_utils
from data.stochastic_focal_patching import StochasticFocalPatchSampler

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

        patch_size = conf.dataset.patch_size
        focal_region_proportion = conf.dataset.focal_region_proportion
        self.patch_sampler = StochasticFocalPatchSampler(patch_size, focal_region_proportion)

    def __getitem__(self, index):
        index_CBCT = index % self.num_datapoints_CBCT
        index_CT = random.randint(0, self.num_datapoints_CT - 1)

        path_CBCT = os.path.join(self.paths_CBCT[index_CBCT], 'CT.nrrd')
        path_CT = os.path.join(self.paths_CT[index_CT], 'CT.nrrd')
        
        # load nrrd as SimpleITK objects
        CBCT = sitk_utils.load(path_CBCT)
        CT = sitk_utils.load(path_CT)

        CBCT = sitk_utils.get_tensor(CBCT)
        CT = sitk_utils.get_tensor(CT)

        # remove first and last slice of CBCT due to FOV
        CBCT = CBCT[1:-1]

        # CT has more slices than CBCT, make sure that CT contains
        # a similar size of the scanned body as by truncating it at both ends
        num_extra_slices = CT.shape[0] - CBCT.shape[0]
        start = int(num_extra_slices * 0.4) # these proportion decided heuristically 
        end = int(-num_extra_slices * 0.6)  # by comparing the CBCT and CT images of the dataset
        CT = CT[start:end]        
        
        # Extract patches
        CBCT, CT = self.patch_sampler.get_patch_pair(CBCT, CT) 

        # Limits the lowest and highest HU unit
        CBCT = np.clip(CBCT, self.hu_min, self.hu_max)
        CT = np.clip(CT, self.hu_min, self.hu_max)

        # Normalize Hounsfield units to range [-1,1]
        CBCT = normalize_from_hu(CBCT, self.hu_min, self.hu_max)
        CT = normalize_from_hu(CT, self.hu_min, self.hu_max)

        # Add channel dimension (1 = grayscale)
        CBCT = CBCT.unsqueeze(0)
        CT = CT.unsqueeze(0)

        return {'A': CBCT, 'B': CT}


    def __len__(self):
        return max(self.num_datapoints_CBCT, self.num_datapoints_CT)




