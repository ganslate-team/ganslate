from dataclasses import dataclass, field
from omegaconf import MISSING
from conf.base.generators import BaseGeneratorConfig
from conf.base.discriminators import BaseDiscriminatorConfig


@dataclass
class BaseGANConfig:
    """Base GAN config."""
    is_train:         bool = True
    model:            str = MISSING
    loss_type:        str = "lsgan"
    norm_type:        str = "instance"
    weight_init_type: str = "normal"
    weight_init_gain: float = 0.02

    generator:         BaseGeneratorConfig = MISSING
    discriminator:     BaseDiscriminatorConfig = MISSING

    n_channels_input:  int = 1 # TODO: think if necessary, probably not
    n_channels_output: int = 1


@dataclass
class Cycle3DGANConfig(BaseGANConfig):
    model:            str = "unpaired_revgan3d"
    generator:         BaseGeneratorConfig = MISSING
    discriminator:     BaseDiscriminatorConfig = MISSING