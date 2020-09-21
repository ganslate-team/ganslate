from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf.base_configs import *


@dataclass
class OptimizerConfig:
    beta1:           float = 0.5
    lr_D:            float = 0.0002
    lr_G:            float = 0.0002
    lambda_A:        float = 10.0
    lambda_B:        float = 10.0
    lambda_identity: float = 0
    lambda_inverse:  float = 0
    proportion_ssim: float = 0.84


@dataclass
class LoggingConfig:
    #experiment_name:  str = now() # Name of the experiment. [Default: current date and time] 
    checkpoint_dir:  str = "./checkpoints/" + "nesto" # TODO: make it datatime. make sure it work in distributed mode
    #inference_dir:   Optional[str] = None  # Path where the inference will store the results
    log_freq:        int = 20
    checkpoint_freq: int = 50
    wandb:           bool = False
    tensorboard:     bool = False


@dataclass
class LoadCheckpointConfig:
    iter:             str = MISSING  # Which iteration's checkpoint to load. [Default: "latest"]
    count_start_iter: int = 1  # Continue the count of epochs from this value. [Default: 1] # TODO: make training not need this by loading the epoch from the checkpoint (?)


@dataclass
class TrainConfig(BaseConfig):
    # TODO: add git hash? will help when re-running or inferencing old runs
    n_iters:         int = MISSING  # Number of iters without linear decay of learning rates. [Default: 200]
    n_iters_decay:   int = MISSING  # Number of last iters in which the learning rates are linearly decayed. [Default: 50]
    
    # gan and generator already specified in BaseConfig as these are used in inference too
    discriminator:   BaseDiscriminatorConfig = MISSING
    
    optimizer:       OptimizerConfig = OptimizerConfig()
    logging:         LoggingConfig = LoggingConfig()
    load_checkpoint: Optional[LoadCheckpointConfig] = None

