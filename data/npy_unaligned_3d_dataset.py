import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import random
import torch
import numpy as np
import json
import random


def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def normalize(image, MIN_B=-1024.0, MAX_B=3072.0):
    # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    image = (image - MIN_B) / (MAX_B - MIN_B)
    return 2*image - 1

def random_patch_3d(volume, min_value, patch_size=(64,64,64), threshold=0.6):
    patch_shape = np.array(patch_size)
    volume_shape = np.array(volume.shape[-3:])
    # a patch can have a starting coordinate anywhere from 
    # where it can fit with the defined patch size
    valid_starting_region = volume_shape - patch_shape

    # pick a random starting point in valid region of volume
    z = random.randint(0, valid_starting_region[0])
    x = random.randint(0, valid_starting_region[1])
    y = random.randint(0, valid_starting_region[2])

    # extract the patch from the volume
    patch = volume[z:z+patch_size[0],
                   x:x+patch_size[1],
                   y:y+patch_size[2]]

    # ratio: number of completely black voxels / total number of voxels 
    # E.g. min_value is -1024 (black)
    black_voxels_ratio = np.count_nonzero(patch <= min_value) / patch.numel()
    # if above threshold, find another patch 
    if black_voxels_ratio > threshold:
        return random_patch_3d(volume, min_value, patch_size)
    return patch

class NpyUnaligned3dDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        #self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, 'A')
        self.dir_B = os.path.join(opt.dataroot, 'B')
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # dataset range of values information for normalization
        self.norm_A = load_json(os.path.join(opt.dataroot, 'normalize_A.json'))
        self.norm_B = load_json(os.path.join(opt.dataroot, 'normalize_B.json'))

        print('len(A),len(B)=', self.A_size, self.B_size)

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
        A = random_patch_3d(A, 
                            min_value=self.norm_A["min"], 
                            patch_size=self.opt.patch_size, 
                            threshold=self.opt.threshold_black_voxels)

        B = random_patch_3d(B, 
                            min_value=self.norm_B["min"], 
                            patch_size=self.opt.patch_size, 
                            threshold=self.opt.threshold_black_voxels)

        # normalization
        A = normalize(A, self.norm_A["min"], self.norm_A["max"])
        B = normalize(B, self.norm_B["min"], self.norm_B["max"])

        # reshape so that it contains the channel as well (1 = grayscale)
        A = A.view(1, *A.shape)
        B = B.view(1, *B.shape)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'NpyUnaligned3dDataset'
