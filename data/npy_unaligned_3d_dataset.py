import os.path
from data.base_dataset import BaseDataset, get_transform, make_dataset
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


def focused_random_zxy(zxy, window, valid_region):
    selected_zxy = []
    # for each axis
    for idx in range(len(zxy)):
        # find the lowest and highest position between which to focus for this axis
        min_position = int(zxy[idx] - window[idx]/2)
        max_position = int(zxy[idx] + window[idx]/2)
        # if one of the boundaries of the focus is outside of the possible area to sample from, cap it
        min_position = max(0, min_position)
        max_position = min(max_position, valid_region[idx])
        # edge cases (no pun intended)
        if min_position > max_position:
            selected_zxy.append(max_position)
        # regular
        else:
            selected_zxy.append(random.randint(min_position, max_position))
    return selected_zxy

def random_patch_3d(volume, patch_size=(64,64,64), min_value=-1024, 
                    focus_around_zxy=None, focus_window_to_volume_proportion=None):
    '''
     volume:           whole CT scan (numpy array)
     patch_size:       size of the 3D volume to be extracted from the original volume
     min_value:        used for prevent taking patches that contain too many voxels of that value or lower
     threshold:        defines allowed proportion of voxels in a volume with values lower or equal to min_value
     focus_around_zxy: enables taking a patch from B that is in a similar location to the patch from A
    '''
    patch_shape = np.array(patch_size)
    volume_shape = np.array(volume.shape[-3:])
    # a patch can have a starting coordinate anywhere from where it can fit with the defined patch size
    valid_starting_region = volume_shape - patch_shape

    if focus_around_zxy is None:
        # pick a random starting point in valid region of volume
        z = random.randint(0, valid_starting_region[0])
        x = random.randint(0, valid_starting_region[1])
        y = random.randint(0, valid_starting_region[2])

    else: # take a relative neighbor of patch A in patch B
        # 3D window/neighborhood of focus_around_zxy from which will be randomly selected a new start zxy for B
        focus_window = np.multiply(volume_shape, focus_window_to_volume_proportion).astype(np.int64)
        # the starting position from A is given in relative form (A_start_zxy / A_shape)
        zxy = np.array(focus_around_zxy) * volume_shape  # find start position of A translated in B
        z, x, y = focused_random_zxy(zxy, focus_window, valid_starting_region)
        
    # extract the patch from the volume
    patch = volume[z:z+patch_size[0],
                   x:x+patch_size[1],
                   y:y+patch_size[2]]

    # used only for focus_around_zxy
    relative_zxy = (np.array([z,x,y]) / volume_shape).tolist()
    return patch, relative_zxy

class NpyUnaligned3dDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        self.opt = opt
        #self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, 'A')
        self.dir_B = os.path.join(opt.dataroot, 'B')
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
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
        A, A_zxy = random_patch_3d(A, 
                                  patch_size=self.opt.patch_size,
                                  min_value=self.norm_A["min"])

        B, _ = random_patch_3d(B, 
                               patch_size=self.opt.patch_size,
                               min_value=self.norm_B["min"],
                               focus_around_zxy=A_zxy, 
                               focus_window_to_volume_proportion=self.opt.focus_window)

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
