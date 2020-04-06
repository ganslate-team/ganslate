import os.path
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class NpyAligned2dDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = np.load(AB_path)
        
        A, B = np.split(AB, 2, axis=1)
        # greyscale images, single color channel. E.g. (1,512,512)
        A = A.reshape(1, A.shape[0], A.shape[1])
        B = B.reshape(1, B.shape[0], B.shape[1])
        A = torch.Tensor(A)
        B = torch.Tensor(B)
        
        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)