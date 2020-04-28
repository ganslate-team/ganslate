from typing import Tuple
from dataclasses import dataclass, field
from omegaconf import MISSING
from util.util import now


@dataclass
class ModelConfig:
    is_train:          bool = True
    use_memory_saving: bool = True  # Turn on memory saving for invertible layers. [Default: True]

    model_GAN:         str = "unpaired_revgan3d"   #
    model_G:           str = "vnet"
    model_D:           str = "n_layers"
    n_layers_D:        int = 3
    start_n_filters_G: int = 16                    # Number of filters in first conv layer of generator TODO: what with this? implement in vnet?
    start_n_filters_D: int = 64                    # Number of filters in first conv layer of discriminator
    norm_layer_type:   str = "instance"            # TODO" implement

    weight_init_type:  str = "normal"
    weight_init_gain:  float = 0.02

    n_channels_input:  int = 1 # TODO: think if necessary, probably not
    n_channels_output: int = 1

    no_lsgan:          bool = False

@dataclass
class DatasetConfig:
    root:         str = "../"
    mode:         str = "dummy"
    direction:    str = "AtoB"      # remove
    
    pool_size:    int = 50
    patch_size:   Tuple[int] = field(default_factory=lambda: (32, 32, 32))
    focal_region_proportion: float = 0.2    # Proportion of focal region size compared to original volume size

    shuffle:      bool = True
    num_workers:  int = 4

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
    experiment_name:  str = now()     # Name of the experiment. [Default: current date and time] TODO: not working in distributed mode
    checkpoints_dir:  str = "./checkpoints"
    print_freq:       int = 50
    save_epoch_freq:  int = 25
    wandb:            bool = False

@dataclass
class Config:
    batch_size:      int = 1
    n_epochs:        int = 200       # Number of epochs without linear decay of learning rates. [Default: 200]
    n_epochs_decay:  int = 50        # Number of last epoch in which the learning rates are linearly decayed. [Default: 50]
    use_cuda:        bool = True     # Use CUDA i.e. GPU(s). [Default: True]

    # Distributed and mixed precision
    distributed:     bool = False
    local_rank:      int = 0
    mixed_precision: bool = False
    opt_level:       str = "O1"
    per_loss_scale:  bool = True

    # Continuing training
    continue_train:  bool = False    # Continue training by loading a checkpoint. [Default: False]
    load_epoch:      str = "latest"  # Which epoch's checkpoint to load. [Default: "latest"]
    continue_epoch:  int = 1         # Continue the count of epochs from this value. [Default: 1] 

    dataset:         DatasetConfig = DatasetConfig()
    model:           ModelConfig = ModelConfig()
    optimizer:       OptimizerConfig = OptimizerConfig()
    logging:         LoggingConfig = LoggingConfig()