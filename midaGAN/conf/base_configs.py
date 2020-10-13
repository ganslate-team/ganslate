from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class BaseDatasetConfig:
    name:         str = MISSING # TODO: used for importing data/name_dataset.py, any better way?
    root:         str = MISSING
    shuffle:      bool = True
    num_workers:  int = 4


@dataclass
class BaseGANConfig:
    """Base GAN config."""
    is_train:         bool = True
    name:             str = MISSING
    loss_type:        str = "lsgan"
    norm_type:        str = "instance"
    weight_init_type: str = "normal"
    weight_init_gain: float = 0.02
    pool_size:        int = 50


@dataclass
class BaseDiscriminatorConfig:
    name:        str = MISSING
    in_channels: int = MISSING


@dataclass
class BaseGeneratorConfig:
    name:        str = MISSING
    in_channels: int = MISSING # TODO: put in GAN config since both G and D use `in_channels`?


@dataclass 
class BaseConfig:
    batch_size:      int = MISSING
    project_dir:     Optional[str] = None  # Needed if project-specific classes are to be imported 

    use_cuda:        bool = True    # Use CUDA i.e. GPU(s). [Default: True]
    mixed_precision: bool = False

    dataset:         BaseDatasetConfig = MISSING
    gan:             BaseGANConfig = MISSING
    generator:       BaseGeneratorConfig = MISSING