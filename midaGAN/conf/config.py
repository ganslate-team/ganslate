from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class BaseDatasetConfig:
    name:         str = MISSING # TODO: used for importing data/name_dataset.py, any better way?
    root:         str = MISSING
    
    pool_size:    int = 50
    shuffle:      bool = True
    num_workers:  int = 4


@dataclass
class BaseGANConfig:
    """Base GAN config."""
    is_train:         bool = True
    name:            str = MISSING
    loss_type:        str = "lsgan"
    norm_type:        str = "instance"
    weight_init_type: str = "normal"
    weight_init_gain: float = 0.02
    # n_channels_input:  int = 1  needed only for 2D approaches
    # n_channels_output: int = 1


@dataclass
class BaseDiscriminatorConfig:
    name: str = MISSING


@dataclass
class BaseGeneratorConfig:
    name:             str = MISSING


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
    #checkpoints_dir:  str = "./checkpoints/"
    output_dir:       str = "./checkpoints/" + "nesto" # TODO: make it datatime. make sure it work in distributed mode
    log_freq:         int = 20
    checkpoint_freq:  int = 50
    wandb:            bool = False
    tensorboard:      bool = False

@dataclass
class Config:
    batch_size:     int = MISSING
    n_iters:        int = MISSING   # Number of iters without linear decay of learning rates. [Default: 200]
    n_iters_decay:  int = MISSING   # Number of last iters in which the learning rates are linearly decayed. [Default: 50]
    use_cuda:       bool = True     # Use CUDA i.e. GPU(s). [Default: True]

    # Mixed precision
    mixed_precision: bool = False
    opt_level:       str = "O1"

    # Continuing training
    continue_train:  bool = False    # Continue training by loading a checkpoint. [Default: False]
    load_iter:       str = "latest"  # Which iteration's checkpoint to load. [Default: "latest"]
    continue_iter:   int = 1         # Continue the count of epochs from this value. [Default: 1] # TODO: make training not need this by loading the epoch from the checkpoint (?)

    dataset:         BaseDatasetConfig = MISSING
    gan:             BaseGANConfig = MISSING
    generator:       BaseGeneratorConfig = MISSING
    discriminator:   BaseDiscriminatorConfig = MISSING

    optimizer:       OptimizerConfig = OptimizerConfig()
    logging:         LoggingConfig = LoggingConfig()