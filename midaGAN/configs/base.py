from typing import Optional
from dataclasses import dataclass
from omegaconf import MISSING


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
    # freq_loss: Optional[FreqLossConfig] = None


@dataclass
class BaseDatasetConfig:
    name: str = MISSING
    root: str = MISSING
    shuffle: bool = True
    num_workers: int = 4


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
    discriminator: Optional[
        BaseDiscriminatorConfig] = None  # discriminator optional as it is not used in inference


@dataclass
class BaseConfig:
    is_train: bool = MISSING

    batch_size: int = MISSING
    project_dir: Optional[str] = None  # Needed if project-specific classes are to be imported

    use_cuda: bool = True  # Use CUDA i.e. GPU(s). [Default: True]
    # Mixed precision
    mixed_precision: bool = False
    opt_level: str = "O1"

    dataset: BaseDatasetConfig = MISSING
    gan: BaseGANConfig = MISSING
