
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf.base_configs import *


@dataclass
class LoggingConfig:
    inference_dir:  str = MISSING  # Path where the inference will store the results
    checkpoint_dir: str = MISSING  # Where the checkpoints and training config yaml were saved


@dataclass
class LoadCheckpointConfig:
    iter: str = MISSING 


@dataclass
class SlidingWindowConfig:
    window_size: Tuple[int] = MISSING
    batch_size:  int = 1
    overlap:     float = 0.25
    mode:        str = 'gaussian' # 'constant' or 'gaussian', https://docs.monai.io/en/latest/inferers.html#monai.inferers.SlidingWindowInferer


@dataclass
class InferenceConfig(BaseConfig):
    load_checkpoint: LoadCheckpointConfig = LoadCheckpointConfig()
    logging:         LoggingConfig = LoggingConfig()
    sliding_window:  Optional[SlidingWindowConfig] = None


