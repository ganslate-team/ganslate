from typing import Optional
from dataclasses import dataclass

from ganslate.configs import base, validation_testing


@dataclass
class InferenceConfig(base.BaseEngineConfig):
    is_deployment: bool = False
    dataset: Optional[base.BaseDatasetConfig] = None
    sliding_window: Optional[validation_testing.SlidingWindowConfig] = None
    checkpointing: base.CheckpointingConfig = base.CheckpointingConfig()
