from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from util.file_utils import now

from conf.base.datasets import BaseDatasetConfig
from conf.base.gans import BaseGANConfig

@dataclass
class OptimizerConfig:
    beta1:           float = 0.5
    lr_D:            float = 0.0002
    lr_G:            float = 0.0002
    lambda_A:        float = 10.0
    lambda_B:        float = 10.0
    lambda_identity: float = 0.1
    lambda_inverse:  float = 0.05
    proportion_ssim: float = 0.84


@dataclass
class LoggingConfig:
    #experiment_name:  str = now() # Name of the experiment. [Default: current date and time] TODO: not working in distributed mode
    #checkpoints_dir:  str = "./checkpoints/"
    output_dir:       str = "./checkpoints/" + "nesto" #now()
    print_freq:       int = 50
    checkpoint_freq:  int = 5
    wandb:            bool = False

@dataclass
class Config:
    batch_size:      int = MISSING
    n_iters:        int = MISSING       # Number of iters without linear decay of learning rates. [Default: 200]
    n_iters_decay:  int = MISSING        # Number of last iters in which the learning rates are linearly decayed. [Default: 50]
    use_cuda:        bool = True     # Use CUDA i.e. GPU(s). [Default: True]

    # Distributed and mixed precision
    distributed:     bool = False
    mixed_precision: bool = False
    opt_level:       str = "O1"
    per_loss_scale:  bool = True

    # Continuing training
    continue_train:  bool = False    # Continue training by loading a checkpoint. [Default: False]
    load_iter:      str = "latest"  # Which iteration's checkpoint to load. [Default: "latest"]
    continue_iter:  int = 1         # Continue the count of epochs from this value. [Default: 1] # TODO: make training not need this

    dataset:         BaseDatasetConfig = MISSING
    gan:             BaseGANConfig = MISSING
    optimizer:       OptimizerConfig = OptimizerConfig()
    logging:         LoggingConfig = LoggingConfig()