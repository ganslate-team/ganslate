from typing import Optional
from dataclasses import dataclass

from midaGAN.configs import base, evaluation


@dataclass
class InferenceConfig(base.BaseEngineConfig):
    is_deployment: bool = False
    dataset: Optional[base.BaseDatasetConfig] = None
    sliding_window: Optional[evaluation.SlidingWindowConfig] = None
    checkpointing: base.CheckpointingConfig = base.CheckpointingConfig()
