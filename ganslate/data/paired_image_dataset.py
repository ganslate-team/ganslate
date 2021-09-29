import random
from pathlib import Path
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset

from ganslate.data.utils.transforms import get_paired_image_transform
from ganslate.utils.io import make_dataset_of_files

# Config imports
from dataclasses import dataclass, field
from ganslate import configs


EXTENSIONS = ['.jpg', '.jpeg', '.png']


@dataclass
class PairedImageDatasetConfig(configs.base.BaseDatasetConfig):
    image_channels: int = 3
    # Preprocessing instructions for images at load time:
    #   Initial resizing:   'resize', 'scale_width'
    #   Random transforms:  'random_zoom', 'random_crop', 'random_flip'
    # Note: During val/test, make sure to not include random transforms in the YAML config 
    preprocess: Tuple[str] = ('resize', 'random_crop', 'random_flip')
    # Sizes in (H, W) format
    load_size: Tuple[int, int] = field(default_factory=lambda: [286, 572])
    final_size: Tuple[int, int] = field(default_factory=lambda: [256, 512])


class PairedImageDataset(Dataset):

    def __init__(self, conf):

        self.dir_A = Path(conf[conf.mode].dataset.root) / 'A'
        self.dir_B = Path(conf[conf.mode].dataset.root) / 'B'

        self.A_paths = make_dataset_of_files(self.dir_A, EXTENSIONS)
        self.B_paths = make_dataset_of_files(self.dir_B, EXTENSIONS)
        self.n_samples = len(self.A_paths)

        self.transform = get_paired_image_transform(conf)
        self.rgb_or_grayscale = 'RGB' if conf[conf.mode].dataset.image_channels == 3 else 'L'

    def __getitem__(self, index):
        index = index % self.n_samples 

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A_img = Image.open(A_path).convert(self.rgb_or_grayscale)
        B_img = Image.open(B_path).convert(self.rgb_or_grayscale)

        A, B = self.transform(A_img, B_img)

        return {'A': A, 'B': B}

    def __len__(self):
        return self.n_samples
