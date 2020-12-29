from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf.base_configs import *
from midaGAN.conf.common_configs import *

from midaGAN.conf.eval_config import *


@dataclass
class LossMaskingConfig:
    masking_value: float = -1
    operator: str = "eq"


@dataclass
class ImageFilterConfig:  # Filtering for images uploaded to wandb to show windowing
    min: float = -1
    max: float = 1


@dataclass
class LoggingConfig:
    #experiment_name:  str = now() # Name of the experiment. [Default: current date and time]
    checkpoint_dir: str = "./checkpoints/" + "nesto"  # TODO: make it datatime. make sure it work in distributed mode
    log_freq: int = 50
    checkpoint_freq: int = 2000
    tensorboard: bool = False
    wandb: Optional[WandbConfig] = None


@dataclass
class LoadCheckpointConfig:
    iter: str = MISSING  # Which iteration's checkpoint to load.
    count_start_iter: int = 1  # Continue the count of epochs from this value. [Default: 1] # TODO: make training not need this by loading the epoch from the checkpoint (?)
    reset_optimizers: bool = False  # If true, the checkpoint optimizer state_dict won't be loaded and optimizers will start from scratch.


@dataclass
class TrainMetricConfig:
    output_distributions_D: bool = False
    ssim: bool = False


@dataclass
class TrainConfig(BaseConfig):
    is_train: bool = True

    # TODO: add git hash? will help when re-running or inferencing old runs
    n_iters: int = MISSING  # Number of iters without linear decay of learning rates. [Default: 200]
    n_iters_decay: int = MISSING  # Number of last iters in which the learning rates are linearly decayed. [Default: 50]

    logging: LoggingConfig = LoggingConfig()
    load_checkpoint: Optional[LoadCheckpointConfig] = None
    seed: Optional[int] = None  # Seed for reproducibility
    metrics: TrainMetricConfig = TrainMetricConfig()  # Metrics to log while training!

    # Separate evaluation config that will be run with a full-volume dataloader.
    # Can be used for intermittent SSIM, dose calcs etc
    evaluation: Optional[EvalConfig] = None
