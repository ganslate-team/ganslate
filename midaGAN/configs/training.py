from typing import Optional
from dataclasses import dataclass
from omegaconf import MISSING
from midaGAN import configs


@dataclass
class TrainMetricConfig:
    output_distributions_D: bool = False
    ssim: bool = False


@dataclass
class TrainConfig(configs.base.BaseEngineConfig):
    # TODO: add git hash? will help when re-running or inferencing old runs

    # Number of iters without linear decay of learning rates.
    n_iters: int = MISSING
    # Number of last iters in which the learning rates are linearly decayed.
    n_iters_decay: int = MISSING

    gan: configs.base.BaseGANConfig = MISSING
    # Iteration number of the checkpoint to load
    load_checkpoint: Optional[int] = None
    # If false, the saved optimizers won't be loaded and the optimizers will start from scratch
    load_optimizers: Optional[bool] = None

    seed: Optional[int] = None  # Seed for reproducibility
    metrics: TrainMetricConfig = TrainMetricConfig()  # Metrics to log while training!

    # Separate validation config that will be run with a full-volume dataloader.
    # Can be used for intermittent SSIM
    validation: Optional[configs.evaluation.ValidationConfig] = None
