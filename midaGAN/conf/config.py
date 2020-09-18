from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class BaseDatasetConfig:
    name:         str = MISSING # TODO: used for importing data/name_dataset.py, any better way?
    root:         str = MISSING
    shuffle:      bool = True
    num_workers:  int = 4


@dataclass
class BaseGANConfig:
    """Base GAN config."""
    is_train:         bool = True
    name:             str = MISSING
    loss_type:        str = "lsgan"
    norm_type:        str = "instance"
    weight_init_type: str = "normal"
    weight_init_gain: float = 0.02
    pool_size:        int = 50
    # n_channels_input:  int = 1  needed only for 2D approaches
    # n_channels_output: int = 1


@dataclass
class BaseDiscriminatorConfig:
    name:        str = MISSING
    in_channels: int = MISSING


@dataclass
class BaseGeneratorConfig:
    name:        str = MISSING
    in_channels: int = MISSING # TODO: put in GAN config since both G and D use `in_channels`?


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
    inference_dir:   Optional[str] = None  # Path where the inference will store the results
    log_freq:        int = 20
    checkpoint_freq: int = 50
    wandb:           bool = False
    tensorboard:     bool = False


@dataclass
class Config:
    # TODO: add git hash? will help when re-running or inferencing old runs
    project_dir:     Optional[str] = None  # Needed if project-specific classes are to be imported 
    batch_size:      int = MISSING   
    n_iters:         int = MISSING  # Number of iters without linear decay of learning rates. [Default: 200]
    n_iters_decay:   int = MISSING  # Number of last iters in which the learning rates are linearly decayed. [Default: 50]
    use_cuda:        bool = True    # Use CUDA i.e. GPU(s). [Default: True]

    # Mixed precision
    mixed_precision: bool = False
    opt_level:       str = "O1"

    # Continuing training
    continue_train:  bool = False  # Continue training by loading a checkpoint. [Default: False]
    load_iter:       Optional[str] = None  # Which iteration's checkpoint to load. [Default: "latest"]
    continue_iter:   int = 1  # Continue the count of epochs from this value. [Default: 1] # TODO: make training not need this by loading the epoch from the checkpoint (?)

    dataset:         BaseDatasetConfig = MISSING
    gan:             BaseGANConfig = MISSING
    generator:       BaseGeneratorConfig = MISSING
    discriminator:   BaseDiscriminatorConfig = MISSING

    optimizer:       OptimizerConfig = OptimizerConfig()
    logging:         LoggingConfig = LoggingConfig()


@dataclass
class InferenceConfig:
    inference_dir:  str = MISSING
    checkpoint_dir: str = MISSING
    load_iter:      str = MISSING
    dataset:        Dict[str, Any] = MISSING  # Type checked in Config