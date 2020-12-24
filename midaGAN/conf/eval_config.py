
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf.base_configs import *
from midaGAN.conf.common_configs import *

@dataclass
class LoggingConfig:
    inference_dir:  str = MISSING  # Path where the inference will store the results
    tensorboard:     bool = False
    wandb:           Optional[WandbConfig] = None

@dataclass
class MetricConfig:
    ssim:                       bool = True
    psnr:                       bool = True
    nmse:                       bool = True
    mse:                        bool = True

    hu_accuracy:                bool = False
    dosimetric_calculations:    bool = False

@dataclass
class SlidingWindowConfig:
    window_size: Tuple[int] = MISSING
    batch_size:  int = 1
    overlap:     float = 0.25
    mode:        str = 'gaussian' # 'constant' or 'gaussian', https://docs.monai.io/en/latest/inferers.html#monai.inferers.SlidingWindowInferer

@dataclass
class EvalConfig:
    is_train:        bool = False # Training mode is False for framework
    batch_size:      int = 1
    freq:            int = 1 # Every n iterations to run eval
    metrics:         MetricConfig = MetricConfig()
    samples:         int = 4 # Number of samples from the data to run evaluation for
    sliding_window:  Optional[SlidingWindowConfig] = None
    logging:         LoggingConfig = LoggingConfig()
    dataset:         BaseDatasetConfig = MISSING
