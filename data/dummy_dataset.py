import os.path
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class DummyDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))

    def __getitem__(self, index):
        A = torch.rand(1, 32, 32, 32)
        B = torch.rand(1, 32, 32, 32)

        return {'A': A, 'B': B,

                'A_paths': True, 'B_paths': False}
                #'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return 1000 #len(self.AB_paths)

    def name(self):
        return 'DummyDataset'
