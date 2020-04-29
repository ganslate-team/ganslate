from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class BaseGeneratorConfig:
    model:             str = MISSING
    start_n_filters:   int = MISSING
    
@dataclass
class PiVnetGeneratorConfig(BaseGeneratorConfig):
    """Partially-invertible V-Net generator."""
    model:             str = "pi_vnet"
    start_n_filters:   int = 16
    use_memory_saving: bool = True  # Turn on memory saving for invertible layers. [Default: True]
