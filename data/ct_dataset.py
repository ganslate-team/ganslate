import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from util.file_utils import make_dataset_of_files, load_json
from util.preprocessing import normalize_from_hu
from data.stochastic_focal_patching import StochasticFocalPatchSampler


EXTENSIONS = ['.npy']

class CTDataset(Dataset):
    def __init__(self, conf):
        dir_A = os.path.join(conf.dataset.root, 'A')
        dir_B = os.path.join(conf.dataset.root, 'B')
        self.A_paths = make_dataset_of_files(dir_A, EXTENSIONS)
        self.B_paths = make_dataset_of_files(dir_B, EXTENSIONS)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # Dataset range of values information for normalization
        norm_A = os.path.join(conf.dataset.root, 'normalize_A.json')
        norm_B = os.path.join(conf.dataset.root, 'normalize_B.json')
        self.norm_A = load_json(norm_A)
        self.norm_B = load_json(norm_B)

        patch_size = conf.dataset.patch_size
        focal_region_proportion = conf.dataset.focal_region_proportion
        self.patch_sampler = StochasticFocalPatchSampler(patch_size, focal_region_proportion)

    def __getitem__(self, index):
        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)

        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]
        
        A = torch.Tensor(np.load(A_path))
        B = torch.Tensor(np.load(B_path))
        
        A, B = self.patch_sampler.get_patch_pair(A, B) # Extract patches

        # Normalize Hounsfield units to range [-1,1]
        A = normalize_from_hu(A, self.norm_A["min"], self.norm_A["max"])
        B = normalize_from_hu(B, self.norm_B["min"], self.norm_B["max"])

        # Add channel dimension (1 = grayscale)
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)

        return {'A': A, 'B': B}


    def __len__(self):
        return max(self.A_size, self.B_size)




