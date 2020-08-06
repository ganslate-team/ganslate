from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf.generators import BaseGeneratorConfig
from midaGAN.conf.discriminators import BaseDiscriminatorConfig


@dataclass
class BaseGANConfig:
    """Base GAN config."""
    is_train:         bool = True
    is_3d:            bool = True       # if True, use 3D GAN, otherwise 2D
    model:            str = MISSING
    loss_type:        str = "lsgan"
    norm_type:        str = "instance"
    weight_init_type: str = "normal"
    weight_init_gain: float = 0.02

    generator:         BaseGeneratorConfig = MISSING
    discriminator:     BaseDiscriminatorConfig = MISSING

    # n_channels_input:  int = 1  needed only for 2D approaches
    # n_channels_output: int = 1

@dataclass
class CycleGANConfig(BaseGANConfig):
    """CycleGAN"""
    model: str = "cyclegan"

@dataclass
class PiCycleGANConfig(BaseGANConfig):
    """Partially-invertible CycleGAN"""
    model: str = "picyclegan"