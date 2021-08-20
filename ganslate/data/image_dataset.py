import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from ganslate.data.utils.transforms import get_transform
from ganslate.utils.io import make_dataset_of_files

# Config imports
from dataclasses import dataclass
from ganslate import configs


@dataclass
class ImageDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "ImageDataset"
    image_channels: int = 3
    # Scaling and cropping of images at load time:
    # [resize_and_crop | crop | scale_width | scale_width_and_crop | none]'
    preprocess: str = "resize_and_crop"
    load_size: int = 286
    crop_size: int = 256
    flip: bool = True


EXTENSIONS = ['.jpg', '.jpeg', '.png']


class ImageDataset(Dataset):

    def __init__(self, conf):

        self.dir_A = Path(conf[conf.mode].dataset.root) / 'A'
        self.dir_B = Path(conf[conf.mode].dataset.root) / 'B'

        self.A_paths = make_dataset_of_files(self.dir_A, EXTENSIONS)
        self.B_paths = make_dataset_of_files(self.dir_B, EXTENSIONS)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform = get_transform(conf)
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
