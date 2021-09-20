from typing import Optional, Any
from dataclasses import dataclass

from ganslate.configs.training import TrainConfig
from ganslate.configs.validation_testing import ValidationConfig, TestConfig
from ganslate.configs.inference import InferenceConfig


@dataclass
class Config:
    # Enables importing project-specific classes located in the project's dir
    project: Optional[Any] = None
    # Modes handled internally
    mode: str = "train"

    train: TrainConfig = TrainConfig()
    val: Optional[ValidationConfig] = None
    test: Optional[TestConfig] = None
    infer: Optional[InferenceConfig] = None
