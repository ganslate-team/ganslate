import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from util.util import make_dataset, load_json
from util.preprocessing import normalize_from_hu
from data.focal_random_patch import focal_random_patch
EXTENSIONS = ['.npy']


class NpyUnaligned3dDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        #self.root = opt.dataroot
        dir_A = os.path.join(opt.dataroot, 'A')
        dir_B = os.path.join(opt.dataroot, 'B')
        self.A_paths = sorted( make_dataset(self.dir_A, EXTENSIONS) )
        self.B_paths = sorted( make_dataset(self.dir_B, EXTENSIONS) )
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # dataset range of values information for normalization
        # TODO: make it elegant
        self.norm_A = load_json(os.path.join(opt.dataroot, 'normalize_A.json'))
        self.norm_B = load_json(os.path.join(opt.dataroot, 'normalize_B.json'))

    def __getitem__(self, index):
        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)

        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]
        
        A = np.load(A_path)
        B = np.load(B_path)

        A = torch.Tensor(A)
        B = torch.Tensor(B)

        # random patch extraction
        A, A_zxy = focal_random_patch(volume=A, 
                                      patch_size=self.opt.patch_size)

        B, _ = focal_random_patch(volume=B, 
                                  patch_size=self.opt.patch_size,
                                  focus_around_zxy=A_zxy, 
                                  focus_window_to_volume_proportion=self.opt.focus_window)

        # normalize Hounsfield units to range [-1,1]
        A = normalize_from_hu(A, self.norm_A["min"], self.norm_A["max"])
        B = normalize_from_hu(B, self.norm_B["min"], self.norm_B["max"])

        # reshape so that it contains the channel as well (1 = grayscale)
        A = A.view(1, *A.shape)
        B = B.view(1, *B.shape)

        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.A_size, self.B_size)




