from typing import Optional
from dataclasses import dataclass

from midaGAN.configs import base, evaluation


@dataclass
class InferenceConfig(base.BaseEngineConfig):
    sliding_window: Optional[evaluation.SlidingWindowConfig] = None
    checkpointing: base.CheckpointingConfig = base.CheckpointingConfig()
