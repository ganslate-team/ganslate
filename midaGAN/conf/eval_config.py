
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING
<<<<<<< HEAD
from midaGAN.conf.inference_config import *
from midaGAN.conf.base_configs import *

@dataclass
class MetricConfig:
    ssim:                       bool = False
=======
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

>>>>>>> e12b23b0ddd49bb52b6c7ca61b1a58a1204be971
    hu_accuracy:                bool = False
    dosimetric_calculations:    bool = False

@dataclass
class SlidingWindowConfig:
    window_size: Tuple[int] = MISSING
    batch_size:  int = 1
    overlap:     float = 0.25
    mode:        str = 'gaussian' # 'constant' or 'gaussian', https://docs.monai.io/en/latest/inferers.html#monai.inferers.SlidingWindowInferer

<<<<<<< HEAD
@dataclass
class EvalConfig(BaseConfig):
    enable:          bool = False # Enable eval mode
    eval_freq:       int = 1 # Every n iterations to run eval
    metrics:         MetricConfig = MetricConfig()
    samples:         int = 4 # Number of samples from the data to run evaluation for
    sliding_window:  Optional[SlidingWindowConfig] = None
    
    logging:         Any = None
    gan:             Optional[BaseGANConfig] = None
    generator:       Optional[BaseGeneratorConfig] = None
=======

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
>>>>>>> e12b23b0ddd49bb52b6c7ca61b1a58a1204be971
