from typing import Optional
from dataclasses import dataclass


@dataclass
class ImageFilterConfig:  # Filtering for images uploaded to wandb to show windowing
    min: float = -1
    max: float = 1


@dataclass
class WandbConfig:
    project: str = "my-project"
    entity: Optional[str] = None
    image_filter: Optional[ImageFilterConfig] = None
    run: Optional[str] = None
