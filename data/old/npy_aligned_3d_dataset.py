import os.path
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class NpyAligned3dDataset(BaseDataset):
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
        # the loaded array should be a tuple of two volumes
        A, B = AB[0], AB[1]
        A = torch.Tensor(A)
        B = torch.Tensor(B)
        # reshape so that it contains the channel as well (1 = grayscale)
        A = A.view(1, *A.shape)
        B = B.view(1, *B.shape)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)