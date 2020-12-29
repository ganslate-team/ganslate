from typing import Optional, Tuple
from dataclasses import dataclass
from omegaconf import MISSING
from midaGAN.configs import base_configs, common_configs


@dataclass
class LoggingConfig:
    inference_dir: str = MISSING  # Path where the inference will store the results
    tensorboard: bool = False
    wandb: Optional[common_configs.WandbConfig] = None


@dataclass
class MetricConfig:
    ssim: bool = True
    psnr: bool = True
    nmse: bool = True
    mse: bool = True

    hu_accuracy: bool = False
    dosimetric_calculations: bool = False


@dataclass
class SlidingWindowConfig:
    # https://docs.monai.io/en/latest/inferers.html#monai.inferers.SlidingWindowInferer
    window_size: Tuple[int] = MISSING
    batch_size: int = 1
    overlap: float = 0.25
    mode: str = 'gaussian'


@dataclass
class EvalConfig:
    project_dir: Optional[str] = None  # Needed if project-specific classes are to be imported
    is_train: bool = False  # Training mode is False for framework
    batch_size: int = 1
    freq: int = 1  # Every n iterations to run eval
    metrics: MetricConfig = MetricConfig()
    samples: int = 4  # Number of samples from the data to run evaluation for
    sliding_window: Optional[SlidingWindowConfig] = None
    logging: LoggingConfig = LoggingConfig()
    dataset: base_configs.BaseDatasetConfig = MISSING
