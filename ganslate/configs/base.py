from typing import Optional, Tuple
from dataclasses import dataclass
from omegaconf import MISSING, II

############################### Dataset ########################################


@dataclass
class BaseDatasetConfig:
    _target_: str = MISSING
    root: str = MISSING
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
class GeneratorInOutChannelsConfig:
    AB: Tuple[int, int] = MISSING
    BA: Optional[Tuple[int, int]] = II("train.gan.generator.in_out_channels.AB")

@dataclass
class BaseGeneratorConfig:
    _target_: str = MISSING
    # TODO: When OmegaConf implements Union, enable entering a single int when only AB is needed,
    # or when AB and BA are the same. Otherwise use the GeneratorInOutChannelsConfig.
    in_out_channels: GeneratorInOutChannelsConfig = GeneratorInOutChannelsConfig
        
@dataclass
class DiscriminatorInChannelsConfig:
    B: int = MISSING
    A: Optional[int] = II("train.gan.discriminator.in_channels.B")

@dataclass
class BaseDiscriminatorConfig:
    _target_: str = MISSING
    # TODO: When OmegaConf implements Union, enable entering a single int when only B is needed,
    # or when B and A are the same. Otherwise use the DiscriminatorInChannelsConfig.
    in_channels: DiscriminatorInChannelsConfig = DiscriminatorInChannelsConfig

@dataclass
class BaseGANConfig:
    """Base GAN config."""
    _target_: str = MISSING
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
    project: str = "ganslate-project"
    entity: Optional[str] = None
    # Name of the wandb run
    run: Optional[str] = None
    # Run id to be provided incase runs are to be resumed
    id: Optional[str] = None


@dataclass
class CheckpointingConfig:
    # Iteration number of the checkpoint to load [for continuing training or test/val/infer]
    load_iter: int = MISSING


@dataclass
class MultiModalitySplitConfig:
    # Allows logging of multi-modality images by splitting them over channel dimension accordingly.
    # For example, if tensor `real_A` has 4 channels and contains 2 image modalities, where they
    # have 1 and 3 channels respectively, then `A` attribute needs to be specified as [1, 3].
    # For default cases of single grayscale or RGB images, this config needs not be defined.
    A: Optional[Tuple[int]] = None
    B: Optional[Tuple[int]] = None


@dataclass
class LoggingConfig:
    # How often (in iters) to log during *training* [Not used in other modes]
    freq: int = 50
    # Specifies how to split multi modality images for logging purposes.
    multi_modality_split: Optional[MultiModalitySplitConfig] = None
    # Use Tensorboard?
    tensorboard: bool = False
    # Use Weights & Biases?
    wandb: Optional[WandbConfig] = None
    # Optionally, log windowed images with the min and max values for windowing specified here
    image_window: Optional[Tuple[float, float]] = None


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
