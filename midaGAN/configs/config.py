from typing import Optional
from dataclasses import dataclass

from midaGAN.configs.training import TrainConfig
from midaGAN.configs.evaluation import ValidationConfig, TestConfig


@dataclass
class Config:
    # Enables importing project-specific classes located in the project's dir
    project_dir: Optional[str] = None
    # Modes handled internally
    mode: str = "train"

    train: TrainConfig = TrainConfig()
    val: Optional[ValidationConfig] = None
    test: Optional[TestConfig] = None
