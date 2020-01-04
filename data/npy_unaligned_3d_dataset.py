import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import numpy as np


class NpyUnaligned3dDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print('len(A),len(B)=', self.A_size, self.B_size)

        # TODO: dataset information (min HU, max HU...)
        # by loading normalize.json from the dataroot?

    def __getitem__(self, index):
        index_A = index # % self.A_size
        index_B = random.randint(0, self.B_size - 1)
        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]
        A_img = np.load(A_path)
        B_img = np.load(B_path)

        A = torch.Tensor(A)
        B = torch.Tensor(B)
        # reshape so that it contains the channel as well (1 = grayscale)
        A = A.view(1, *A.shape)
        B = B.view(1, *B.shape)

        # TODO: implement normalization
        # TODO: implement random patch extraction

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'NpyUnaligned3dDataset'
