from typing import List
from dataclasses import dataclass, field

@dataclass
class BaseConfig:

    # BASE
    batch_size:     int = 1
    continue_train: bool = True
    epoch:          str = "latest"
    epoch_count:    int = 1
    niter:          int = 4
    niter_decay:    int = 0
    phase:          str = "train" # TODO: remove
    name:           str = "distmp1"
    gpu_ids:        List[int] = field(default_factory=lambda: []) # replace for is_multigpu
    pool_size:      int = 50

    # DATASET
    dataroot:     str = "../"
    dataset_mode: str = "dummy"
    direction:    str = "AtoB"
    focus_window: float = 0.2
    shuffle:      bool = True
    num_workers:  int = 4
    patch_size:   List[int] = field(default_factory=lambda: [32, 32, 32])

    # MODEL
    model: str = "unpaired_revgan3d"
     
    init_gain:           float = 0.02
    init_type:           str = "normal"
    input_nc:            int = 1
    output_nc:           int = 1
    is_train:            bool = True
    generator_model:     str = "vnet_generator" # TODO: change name to generator_model
    discriminator_model: str = "basic" # TODO: basic unnecessary
    n_layers_D:          int = 3
    use_naive:           bool = False # TODO: rename
    norm:                str = "instance" # TODO" implement
    ndf:                 int = 64
    ngf:                 int = 64

    # OPTIMIZER/SOLVER
    beta1:           float = 0.5
    lr_D:            float = 0.0002
    lr_G:            float = 0.0002
    lr_policy:       str = "lambda"
    lr_decay_iters:  int = 50
    lambda_A:        float = 10.0
    lambda_B:        float = 10.0
    lambda_identity: float = 0.1
    lambda_inverse:  float = 0.05
    proportion_ssim: float = 0.84
    no_lsgan:        bool = False

    # LOGGING
    checkpoints_dir:  str = "./checkpoints"
    no_html:          bool = False
    display_freq:     int = 50
    display_id:       int = -1
    display_ncols:    int = 4
    display_winsize:  int = 256
    print_freq:       int = 50
    save_epoch_freq:  int = 25
    save_latest_freq: int = 5000
    update_html_freq: int = 50
    wandb:            bool = False
    
    distributed: bool = False
    local_rank: int = 0
    mixed_precision: bool = False
    opt_level: str = "O1"
    per_loss_scale: bool = False
    
    