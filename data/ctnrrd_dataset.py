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

class CTNRRDDataset(Dataset):
    def __init__(self, conf):
        dir_A = os.path.join(conf.dataset.root, 'A')
        dir_B = os.path.join(conf.dataset.root, 'B')
        self.A_paths = make_dataset_of_folders(dir_A, EXTENSIONS)
        self.B_paths = make_dataset_of_folders(dir_B, EXTENSIONS)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.dataset.hounsfield_units_range

        patch_size = conf.dataset.patch_size
        focal_region_proportion = conf.dataset.focal_region_proportion
        self.patch_sampler = StochasticFocalPatchSampler(patch_size, focal_region_proportion)

    def __getitem__(self, index):
        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)

        A_path = os.path.join(self.A_paths[index_A], 'CT.nrrd')
        B_path = os.path.join(self.B_paths[index_B], 'CT.nrrd')
        
        # load nrrd as SimpleITK objects
        A = sitk_utils.load(A_path)
        B = sitk_utils.load(B_path)

        A = sitk_utils.get_tensor(A)
        B = sitk_utils.get_tensor(B)
        
        # Extract patches
        A, B = self.patch_sampler.get_patch_pair(A, B) 

        # Limits the lowest and highest HU unit
        A = np.clip(A, self.hu_min, self.hu_max)
        B = np.clip(B, self.hu_min, self.hu_max)

        # Normalize Hounsfield units to range [-1,1]
        A = normalize_from_hu(A, self.hu_min, self.hu_max)
        B = normalize_from_hu(B, self.hu_min, self.hu_max)

        # Add channel dimension (1 = grayscale)
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)

        return {'A': A, 'B': B}


    def __len__(self):
        return max(self.A_size, self.B_size)




