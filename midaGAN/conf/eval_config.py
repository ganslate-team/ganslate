
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf.inference_config import *
from midaGAN.conf.base_configs import *

@dataclass
class MetricConfig:
    ssim:                       bool = False
    hu_accuracy:                bool = False
    dosimetric_calculations:    bool = False

@dataclass
class SlidingWindowConfig:
    window_size: Tuple[int] = MISSING
    batch_size:  int = 1
    overlap:     float = 0.25
    mode:        str = 'gaussian' # 'constant' or 'gaussian', https://docs.monai.io/en/latest/inferers.html#monai.inferers.SlidingWindowInferer

@dataclass
class EvalConfig(BaseConfig):
    enable:          bool = True # Enable eval mode
    eval_freq:       int = 1 # Every n iterations to run eval
    metrics:         MetricConfig = MetricConfig()
    samples:         int = 4 # Number of samples from the data to run evaluation for
    sliding_window:  Optional[SlidingWindowConfig] = None
    
    logging:         Any = None
    gan:             Optional[BaseGANConfig] = None
    generator:       Optional[BaseGeneratorConfig] = None