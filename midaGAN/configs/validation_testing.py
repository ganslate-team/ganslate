from typing import Optional, Tuple, Any, Dict
from dataclasses import dataclass
from omegaconf import MISSING, II
from ganslate.configs import base


@dataclass
class SlidingWindowConfig:
    # https://docs.monai.io/en/latest/inferers.html#monai.inferers.SlidingWindowInferer
    window_size: Tuple[int] = MISSING
    batch_size: int = 1
    overlap: float = 0.25
    mode: str = 'gaussian'


######################## Val and Test Metrics Configs #########################


@dataclass
class BaseValTestMetricsConfig:
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
    # Normalized Mutual Information
    nmi: bool = False
    # Chi-squared Histogram Distance
    histogram_chi2: bool = False


@dataclass
class ValMetricsConfig(BaseValTestMetricsConfig):
    # Set to true if cycle metrics need to be logged (i.e between original and reconstructed image)
    cycle_metrics: bool = True


@dataclass
class TestMetricsConfig(BaseValTestMetricsConfig):
    # True if the metrics comparing input and ground truth are to be computed be as well
    compute_over_input: bool = False
    # Save per image metrics to a CSV for further analysis
    save_to_csv: bool = True


######################## Val and Test General Configs #########################


@dataclass
class BaseValTestConfig(base.BaseEngineConfig):
    sliding_window: Optional[SlidingWindowConfig] = None
    dataset: Optional[base.BaseDatasetConfig] = None
    # Val/test can have multiple datasets provided to it
    multi_dataset: Optional[Dict[str, base.BaseDatasetConfig]] = None


@dataclass
class ValidationConfig(BaseValTestConfig):
    # How frequently to validate (each `freq` iters)
    freq: int = MISSING
    # After which iteration should validation begin
    start_after: int = 0
    metrics: ValMetricsConfig = ValMetricsConfig()


@dataclass
class TestConfig(BaseValTestConfig):
    checkpointing: base.CheckpointingConfig = base.CheckpointingConfig()
    metrics: TestMetricsConfig = TestMetricsConfig()
