from typing import Optional, Tuple
from dataclasses import dataclass
from omegaconf import MISSING
from midaGAN import configs


############################### Dataset ########################################
@dataclass
class BaseDatasetConfig:
    name: str = MISSING
    root: str = MISSING
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True


############ GAN Optimizer, Discriminator, Generator, and Framework #############
@dataclass
class LossMaskingConfig:
    masking_value: float = -1
    operator: str = "eq"


@dataclass
class BaseOptimizerConfig:
    adversarial_loss_type: str = "lsgan"
    beta1: float = 0.5
    beta2: float = 0.999
    lr_D: float = 0.0001
    lr_G: float = 0.0002
    loss_mask: Optional[LossMaskingConfig] = None


@dataclass
class BaseDiscriminatorConfig:
    name: str = MISSING
    in_channels: int = MISSING


@dataclass
class BaseGeneratorConfig:
    name: str = MISSING
    in_channels: int = MISSING


@dataclass
class BaseGANConfig:
    """Base GAN config."""
    name: str = MISSING
    norm_type: str = "instance"
    weight_init_type: str = "normal"
    weight_init_gain: float = 0.02

    optimizer: BaseOptimizerConfig = MISSING
    generator: BaseGeneratorConfig = MISSING
    # Discriminator optional as it is not used in inference
    discriminator: Optional[BaseDiscriminatorConfig] = None

############################### Logging ########################################
@dataclass
class WandbConfig:
    project: str = "my-project"
    entity: Optional[str] = None
    # Min and max value for the image filter # TODO: explain!
    image_filter: Optional[Tuple[float, float]] = None
    run: Optional[str] = None


@dataclass
class LoggingConfig:
    # TODO: make it datetime or smth. make sure it works in distributed mode
    output_dir: str = MISSING
    # How often to log training progress
    log_freq: int = 50
    # How often to save checkpoints
    checkpoint_freq: int = 2000
    tensorboard: bool = False
    wandb: Optional[WandbConfig] = None


############# Config for engines (trainer, tester, inferencer...) ##############
@dataclass
class BaseEngineConfig:
     # Enables importing project-specific classes located in the project's dir
    project_dir: Optional[str] = None

    batch_size: int = MISSING
    use_cuda: bool = True
    mixed_precision: bool = False
    opt_level: str = "O1"

    logging: LoggingConfig = LoggingConfig()

    dataset: BaseDatasetConfig = MISSING

################################################################################
