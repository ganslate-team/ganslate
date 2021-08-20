from typing import Optional
from dataclasses import dataclass
from omegaconf import MISSING
from ganslate.configs import base


@dataclass
class TrainMetricsConfig:
    discriminator_evolution: bool = False
    ssim: bool = False


@dataclass
class TrainCheckpointingConfig(base.CheckpointingConfig):
    # How often (in iters) to save checkpoints during training
    freq: int = 2000
    # After which iteration should checkpointing begin
    start_after: int = 0
    # If False, the saved optimizers won't be loaded when continuing training
    load_optimizers: bool = True
    # Iteration number of the checkpoint to load for continuing training
    load_iter: Optional[int] = None


@dataclass
class TrainConfig(base.BaseEngineConfig):
    # TODO: add git hash? will help when re-running or inferencing old runs

    ################## Overriding defaults of BaseEngineConfig ######################
    output_dir: str = MISSING
    batch_size: int = MISSING
    cuda: bool = True
    mixed_precision: bool = False
    opt_level: str = "O1"
    checkpointing: TrainCheckpointingConfig = TrainCheckpointingConfig()
    logging: base.LoggingConfig = base.LoggingConfig()
    ###########################################################################

    # Number of iters without linear decay of learning rates.
    n_iters: int = MISSING
    # Number of last iters in which the learning rates are linearly decayed.
    n_iters_decay: int = MISSING

    gan: base.BaseGANConfig = MISSING

    seed: Optional[int] = None
    metrics: TrainMetricsConfig = TrainMetricsConfig()
