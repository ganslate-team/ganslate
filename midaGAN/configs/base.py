from typing import Optional, Tuple
from dataclasses import dataclass
from omegaconf import MISSING, II


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
class BaseOptimizerConfig:
    adversarial_loss_type: str = "lsgan"
    beta1: float = 0.5
    beta2: float = 0.999
    lr_D: float = 0.0001
    lr_G: float = 0.0002


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
    project: str = "midaGAN-project"
    entity: Optional[str] = None
    # Name of the wandb run
    run: Optional[str] = None
    # Min and max value for the image filter # TODO: explain!
    image_filter: Optional[Tuple[float, float]] = None


@dataclass
class CheckpointingConfig:
    # Iteration number of the checkpoint to load [for continuing training or test/val/infer]
    load_iter: int = MISSING


@dataclass
class LoggingConfig:
    # How often (in iters) to log during *training* [Not used in other modes]
    freq: int = 50
    # Use Tensorboard?
    tensorboard: bool = False
    # Use Weights & Biases?
    wandb: Optional[WandbConfig] = None


############# Config for engines (trainer, tester, inferencer...) ##############
@dataclass
class BaseEngineConfig:
    """Contains params that all modes need to have, by default they interpolate the value
    of the training config because test/val/infer rely on training's params' values.
    TrainingConfig overrides these defaults for training."""

    # Where the logs and outputs are stored. Modes other than training use it to
    # know where the checkpoints were stored to be able to load them.
    output_dir: str = II("train.output_dir")

    batch_size: int = II("train.batch_size")
    # Uses GPUs if True, otherwise CPU
    cuda: bool = II("train.cuda")
    mixed_precision: bool = II("train.mixed_precision")
    opt_level: str = II("train.opt_level")

    logging: LoggingConfig = II("train.logging")

    dataset: BaseDatasetConfig = MISSING


################################################################################
