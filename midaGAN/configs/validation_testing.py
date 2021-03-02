from typing import Optional, Tuple, Any
from dataclasses import dataclass
from omegaconf import MISSING, II
from midaGAN.configs import base


@dataclass
class ValTestMetricsConfig:
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
class BaseValTestConfig(base.BaseEngineConfig):
    metrics: ValTestMetricsConfig = ValTestMetricsConfig()
    sliding_window: Optional[SlidingWindowConfig] = None

    # Val/test can have multiple datasets provided to it
    # TODO: `Any` used only coz omegaconf doesn't support `Union`
    dataset: Any = MISSING


@dataclass
class ValidationConfig(BaseValTestConfig):
    # How frequently to validate (each `freq` iters)
    freq: int = MISSING
    # After which iteration should validation begin
    start_after: int = 0


@dataclass
class TestConfig(BaseValTestConfig):
    checkpointing: base.CheckpointingConfig = base.CheckpointingConfig()
