import random
from pathlib import Path
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset

from ganslate.data.utils.transforms import get_single_image_transform
from ganslate.utils.io import make_dataset_of_files

# Config imports
from dataclasses import dataclass, field
from ganslate import configs


EXTENSIONS = ['.jpg', '.jpeg', '.png']


@dataclass
class UnpairedImageDatasetConfig(configs.base.BaseDatasetConfig):
    image_channels: int = 3
    # Preprocessing instructions for images at load time:
    #   Initial resizing:   'resize', 'scale_width'
    #   Random transforms:  'random_zoom', 'random_crop', 'random_flip'
    preprocess: Tuple[str] = ('resize', 'random_crop', 'random_flip')
    # Sizes in (H, W) format
    load_size: Tuple[int, int] = field(default_factory=lambda: [286, 286])
    final_size: Tuple[int, int] = field(default_factory=lambda: [256, 256])


class UnpairedImageDataset(Dataset):

    def __init__(self, conf):

        self.dir_A = Path(conf[conf.mode].dataset.root) / 'A'
        self.dir_B = Path(conf[conf.mode].dataset.root) / 'B'

        self.A_paths = make_dataset_of_files(self.dir_A, EXTENSIONS)
        self.B_paths = make_dataset_of_files(self.dir_B, EXTENSIONS)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform = get_single_image_transform(conf)
        self.rgb_or_grayscale = 'RGB' if conf[conf.mode].dataset.image_channels == 3 else 'L'

    def __getitem__(self, index):
        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)

        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert(self.rgb_or_grayscale)
        B_img = Image.open(B_path).convert(self.rgb_or_grayscale)

        A = self.transform(A_img)
        B = self.transform(B_img)

        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.A_size, self.B_size)
