from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf.config import BaseGeneratorConfig


@dataclass
class VnetGeneratorConfig(BaseGeneratorConfig):
    """Partially-invertible V-Net generator."""
    model:             str = "vnet"
    start_n_filters:   int = 16
    use_memory_saving: bool = True  # Turn on memory saving for invertible layers. [Default: True]
    use_inverse:       bool = True  # Specifies if the inverse forward will be used so that it construct the required layers


@dataclass
class Vnet2DGeneratorConfig(BaseGeneratorConfig):
    """Partially-invertible V-Net generator."""
    model:             str = "vnet2d"
    start_n_filters:   int = 16
    use_memory_saving: bool = True  # Turn on memory saving for invertible layers. [Default: True]
    use_inverse:       bool = True  # Specifies if the inverse forward will be used so that it construct the required layers


@dataclass
class UnetGeneratorConfig(BaseGeneratorConfig):
    model:     str='unet'
    num_downs: int=7
    ngf:       int=64


@dataclass
class ResnetGeneratorConfig(BaseGeneratorConfig):
    model:     str='resnet'
    n_blocks:  int=6
    ngf:       int=64