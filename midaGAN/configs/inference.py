from typing import Tuple, Optional
from dataclasses import dataclass
from omegaconf import MISSING
from midaGAN.configs import base_configs


@dataclass
class LoggingConfig:
    inference_dir: str = MISSING  # Path where the inference will store the results
    checkpoint_dir: str = MISSING  # Where the checkpoints and training config yaml were saved


@dataclass
class LoadCheckpointConfig:
    iter: str = MISSING


@dataclass
class SlidingWindowConfig:
    # https://docs.monai.io/en/latest/inferers.html#monai.inferers.SlidingWindowInferer
    window_size: Tuple[int] = MISSING
    batch_size: int = 1
    overlap: float = 0.25
    mode: str = 'gaussian'


@dataclass
class InferenceConfig(base_configs.BaseConfig):
    is_train: bool = False  # Training mode is False for framework
    batch_size: int = 1
    load_checkpoint: LoadCheckpointConfig = LoadCheckpointConfig()
    logging: LoggingConfig = LoggingConfig()
    sliding_window: Optional[SlidingWindowConfig] = None
