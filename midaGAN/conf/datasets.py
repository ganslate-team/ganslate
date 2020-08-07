from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class BaseDatasetConfig:
    name:         str = MISSING # TODO: used for importing data/name_dataset.py, any better way?
    root:         str = MISSING
    
    pool_size:    int = 50
    shuffle:      bool = True
    num_workers:  int = 4

@dataclass
class DummyDatasetConfig(BaseDatasetConfig):
    name:         str = "dummy"

@dataclass
class CTDatasetConfig(BaseDatasetConfig):
    name:         str = "ct"
    patch_size:   Tuple[int] = field(default_factory=lambda: (32, 32, 32))
    focal_region_proportion: float = 0.2    # Proportion of focal region size compared to original volume size


@dataclass
class CBCTtoCTDatasetConfig(BaseDatasetConfig):
    name:                    str = "cbcttoct"
    patch_size:              Tuple[int, int, int] = field(default_factory=lambda: (32, 32, 32))
    hounsfield_units_range:  Tuple[int, int] = field(default_factory=lambda: (-1000, 3000)) #TODO: what should be the default range
    focal_region_proportion: float = 0.2    # Proportion of focal region size compared to original volume size

@dataclass
class BratsDatasetConfig(BaseDatasetConfig):
    name:         str = "brats"
    patch_size:   Tuple[int] = field(default_factory=lambda: (32, 32, 32))
    focal_region_proportion: float = 0    # Proportion of focal region size compared to original volume size

@dataclass
class SliceBasedDatasetConfig(BaseDatasetConfig):
    name:           str = "slice_based"
    image_channels: int = 1  # Number of image channels (1 for grayscale, 3 for RGB)
    pad_or_crop:    str = 'none'

