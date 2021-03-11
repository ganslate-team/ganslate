import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from midaGAN.data.utils.transforms import get_transform
from midaGAN.utils.io import make_dataset_of_files
# Config imports
from dataclasses import dataclass
from midaGAN import configs


@dataclass
class AerialMapsValDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "AerialMapsValDataset"
    image_channels: int = 3
    # Scaling and cropping of images at load time:
    # [resize_and_crop | crop | scale_width | scale_width_and_crop | none]'
    preprocess: str = "resize_and_crop"
    load_size: int = 286
    crop_size: int = 256
    flip: bool = True


EXTENSIONS = ['.jpg', '.jpeg', '.png']


class AerialMapsValDataset(Dataset):

    def __init__(self, conf):

        self.dir_A = Path(conf.train.dataset.root) / 'A'
        self.dir_B = Path(conf.train.dataset.root) / 'B'

        self.A_paths = make_dataset_of_files(self.dir_A, EXTENSIONS)
        self.A_path_dict = {fn.stem.strip('_')[0]:fn for fn in self.A_paths}

        self.B_paths = make_dataset_of_files(self.dir_B, EXTENSIONS)
        self.B_path_dict = {fn.stem.strip('_')[0]:fn for fn in self.B_paths}

        assert self.A_path_dict.keys() == self.B_path_dict.keys()

        self.size = len(self.A_path_dict)

        self.transform = get_transform(conf)
        self.rgb_or_grayscale = 'RGB' if conf.train.dataset.image_channels == 3 else 'L'

    def __getitem__(self, index):
        key = list(self.A_path_dict.keys())[index]
        A_path = self.A_path_dict[key]
        B_path = self.B_path_dict[key]

        A_img = Image.open(A_path).convert(self.rgb_or_grayscale)
        B_img = Image.open(B_path).convert(self.rgb_or_grayscale)

        A = self.transform(A_img)
        B = self.transform(B_img)

        dummy_mask = np.zeros(A.shape)
        dummy_mask[0:A.shape[0]//2, 0:A.shape[1]//2] = 1
        masks = {
            "dummy": dummy_mask
        }

        return {'A': A, 'B': B, "masks": masks}

    def __len__(self):
        return self.size
