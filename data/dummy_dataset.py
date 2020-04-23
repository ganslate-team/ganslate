import os.path
import torch
import numpy as np
from data.base_dataset import BaseDataset, make_dataset


class DummyDataset(BaseDataset):
    '''
    Dummy dataset for quick testing purposes 
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.A_size = 4
        self.B_size = self.A_size

    def __getitem__(self, index):
        #shape = (1, 128, 128, 128)
        shape = (1, 32, 32, 32)
        A = torch.rand(*shape)
        B = torch.rand(*shape)

        return {'A': A, 'B': B,

                'A_paths': True, 'B_paths': False}

    def __len__(self):
        return max(self.A_size, self.B_size)
