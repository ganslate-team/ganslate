from typing import Optional, Tuple
from dataclasses import dataclass
from omegaconf import MISSING, II
from midaGAN import configs


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
class BaseEvaluationConfig:
    # For evaluation ensure that pairing is maintained between the A and B 
    # provided by the attached dataloader

    # To define by the user
    metrics: MetricConfig = MetricConfig()
    sliding_window: Optional[SlidingWindowConfig] = None
    dataset: configs.base.BaseDatasetConfig = MISSING
    batch_size: int = MISSING


@dataclass
class ValidationConfig(BaseEvaluationConfig):
    # How frequently to validate (each `freq` iters)
    freq: int = 1
    # Validation uses the same batch size as training by default
    batch_size: int = II("train.batch_size")


@dataclass
class TestConfig(BaseEvaluationConfig):
    checkpoint_iter: int = 1
