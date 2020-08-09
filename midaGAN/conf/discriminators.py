from dataclasses import dataclass, field
from omegaconf import MISSING

from midaGAN.conf.config import BaseDiscriminatorConfig


@dataclass
class PatchGANDiscriminatorConfig(BaseDiscriminatorConfig):
    model:           str = "patchgan"
    start_n_filters: int = 64
    n_layers:        int = 3

@dataclass
class PatchGAN2DDiscriminatorConfig(BaseDiscriminatorConfig):
    model:           str = "patchgan2d"
    start_n_filters: int = 64
    n_layers:        int = 3