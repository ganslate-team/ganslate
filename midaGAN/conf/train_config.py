from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf.base_configs import *
from midaGAN.conf.eval_config import *


@dataclass
class LossMaskingConfig:
    masking_value:   float = -1
    operator:        str = "eq"

@dataclass
class OptimizerConfig:
    beta1:           float = 0.5
    lr_D:            float = 0.0001
    lr_G:            float = 0.0002
    lambda_A:        float = 10.0
    lambda_B:        float = 10.0
    lambda_identity: float = 0
    lambda_inverse:  float = 0
    proportion_ssim: float = 0.84
    ssim_type:       str = "SSIM" # Possible options are ThreeComponentSSIM, SSIM, MS-SSIM
    loss_mask:       Optional[LossMaskingConfig] = None


@dataclass
class WandbConfig:
    project: str = "my-project"
    entity: Optional[str] = None

@dataclass
class LoggingConfig:
    #experiment_name:  str = now() # Name of the experiment. [Default: current date and time] 
    checkpoint_dir:  str = "./checkpoints/" + "nesto" # TODO: make it datatime. make sure it work in distributed mode
    log_freq:        int = 50
    checkpoint_freq: int = 2000
    tensorboard:     bool = False
    wandb:           Optional[WandbConfig] = None

    
@dataclass
class LoadCheckpointConfig:
    iter:             str = MISSING  # Which iteration's checkpoint to load. 
    count_start_iter: int = 1  # Continue the count of epochs from this value. [Default: 1] # TODO: make training not need this by loading the epoch from the checkpoint (?)
    reset_optimizers: bool = False  # If true, the checkpoint optimizer state_dict won't be loaded and optimizers will start from scratch.


@dataclass
class MetricConfig:
    output_distributions_D:  bool = False
    ssim:                  bool = False


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
    seed:            Optional[int] = None  # Seed for reproducibility
    metrics:         MetricConfig = MetricConfig() # Metrics to log while training! 

    # Separate evaluation config that will be run with a full-volume dataloader. 
    # Can be used for intermittent SSIM, dose calcs etc
    eval:            EvalConfig = EvalConfig() # Evaluation config
