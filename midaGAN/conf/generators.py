from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class BaseGeneratorConfig:
    model:             str = MISSING
    start_n_filters:   int = MISSING
    
@dataclass
class VnetGeneratorConfig(BaseGeneratorConfig):
    """Partially-invertible V-Net generator."""
    model:             str = "vnet"
    start_n_filters:   int = 16
    use_memory_saving: bool = True  # Turn on memory saving for invertible layers. [Default: True]

@dataclass
class Vnet2DGeneratorConfig(BaseGeneratorConfig):
    """Partially-invertible V-Net generator."""
    model:             str = "vnet2d"
    start_n_filters:   int = 16
    use_memory_saving: bool = True  # Turn on memory saving for invertible layers. [Default: True]