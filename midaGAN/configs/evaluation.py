from typing import Optional, Tuple
from dataclasses import dataclass
from omegaconf import MISSING
from midaGAN import configs


@dataclass
class LoggingConfig:
    #inference_dir: str = MISSING  # Path where the inference will store the results
    tensorboard: bool = False
    wandb: Optional[configs.common.WandbConfig] = None


@dataclass
class MetricConfig:
    # SSIM metric between the images
    ssim: bool = True
    # PSNR metric between the images
    psnr: bool = True
    # Normalized MSE
    nmse: bool = True
    # MSE 
    mse: bool = True
    # Abs diff between the two images
    abs_diff: bool = True
    # Set to true if cycle metrics need to be logged
    # i.e A->B->A followed by comparison between the A
    cycle_metrics: bool = False


@dataclass
class SlidingWindowConfig:
    # https://docs.monai.io/en/latest/inferers.html#monai.inferers.SlidingWindowInferer
    window_size: Tuple[int] = MISSING
    batch_size: int = 1
    overlap: float = 0.25
    mode: str = 'gaussian'


@dataclass
class EvaluationConfig:
    # For evaluation ensure that pairing is maintained between the A and B 
    # provided by the attached dataloader

    #project_dir: Optional[str] = None  # Needed if project-specific classes are to be imported
    #is_train: bool = False  # Training mode is False for framework
    #batch_size: int = 1
    freq: int = 1  # Every n iterations to run eval
    metrics: MetricConfig = MetricConfig()
    sliding_window: Optional[SlidingWindowConfig] = None
    #logging: LoggingConfig = LoggingConfig()
    dataset: configs.base.BaseDatasetConfig = MISSING
