from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class ImageFilterConfig:  # Filtering for images uploaded to wandb to show windowing
    min: float = -1
    max: float = 1

@dataclass
class WandbConfig:
    project:      str = "my-project"
    entity:       Optional[str] = None
    run:          Optional[str] = None
    image_filter: Optional[ImageFilterConfig] = None
