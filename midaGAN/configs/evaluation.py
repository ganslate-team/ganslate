from typing import Optional, Tuple
from dataclasses import dataclass
from omegaconf import MISSING, II
from midaGAN.configs import base


@dataclass
class EvalMetricsConfig:
    # SSIM metric between the images
    ssim: bool = True
    # PSNR metric between the images
    psnr: bool = True
    # Normalized MSE
    nmse: bool = True
    # MSE
    mse: bool = True
    # Abs diff between the two images
    mae: bool = True
    # Set to true if cycle metrics need to be logged
    # i.e A->B->A followed by comparison between the A
    cycle_metrics: bool = True


@dataclass
class SlidingWindowConfig:
    # https://docs.monai.io/en/latest/inferers.html#monai.inferers.SlidingWindowInferer
    window_size: Tuple[int] = MISSING
    batch_size: int = 1
    overlap: float = 0.25
    mode: str = 'gaussian'


@dataclass
class BaseEvaluationConfig(base.BaseEngineConfig):
    metrics: EvalMetricsConfig = EvalMetricsConfig()
    sliding_window: Optional[SlidingWindowConfig] = None


@dataclass
class ValidationConfig(BaseEvaluationConfig):
    # How frequently to validate (each `freq` iters)
    freq: int = MISSING
    # How many first iters to skip validation
    start_val_after: int = 0


@dataclass
class TestConfig(BaseEvaluationConfig):
    checkpointing: base.CheckpointingConfig = base.CheckpointingConfig()
