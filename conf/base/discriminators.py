from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class BaseDiscriminatorConfig:
    model:           str = MISSING
    start_n_filters: int = MISSING 

@dataclass
class PatchGANDiscriminatorConfig(BaseDiscriminatorConfig):
    model:           str = "patch_gan"
    start_n_filters: int = 64
    n_layers:        int = 3