from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf.config import BaseGANConfig


@dataclass
class CycleGANConfig(BaseGANConfig):
    """CycleGAN"""
    model: str = "cyclegan"

@dataclass
class PiCycleGANConfig(BaseGANConfig):
    """Partially-invertible CycleGAN"""
    model: str = "picyclegan"