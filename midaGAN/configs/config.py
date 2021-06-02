from typing import Optional, Any
from dataclasses import dataclass

from midaGAN.configs.training import TrainConfig
from midaGAN.configs.validation_testing import ValidationConfig, TestConfig
from midaGAN.configs.inference import InferenceConfig


@dataclass
class Config:
    # Enables importing project-specific classes located in the project's dir
    project_dir: Optional[Any] = None
    # Modes handled internally
    mode: str = "train"

    train: TrainConfig = TrainConfig()
    val: Optional[ValidationConfig] = None
    test: Optional[TestConfig] = None
    infer: Optional[InferenceConfig] = None
