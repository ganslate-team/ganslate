from typing import Tuple, Optional
from dataclasses import dataclass

from midaGAN.configs.evaluation import SlidingWindowConfig


@dataclass
class InferenceConfig:
    batch_size: int
    sliding_window: Optional[SlidingWindowConfig] = None
