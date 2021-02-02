from typing import Optional
from dataclasses import dataclass

from midaGAN.configs.training import TrainConfig
from midaGAN.configs.evaluation import ValidationConfig, TestConfig

@dataclass
class Config:
    mode: str = "train"
    train: TrainConfig = TrainConfig()
    val: Optional[ValidationConfig] = None
    test: Optional[TestConfig] = None