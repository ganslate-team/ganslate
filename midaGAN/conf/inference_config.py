
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf.base_configs import *


@dataclass
class LoggingConfig:
    #experiment_name:  str = now() # Name of the experiment. [Default: current date and time] 
    checkpoint_dir:  str = "./checkpoints/" + "nesto" # TODO: make it datatime. make sure it work in distributed mode
    inference_dir:   Optional[str] = None  # Path where the inference will store the results
    log_freq:        int = 20
    #checkpoint_freq: int = 50
    #wandb:           bool = False
    #tensorboard:     bool = False


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
    logging:        LoggingConfig = LoggingConfig()
    sliding_window: Optional[SlidingWindowConfig] = None


