from typing import Tuple
import torch.nn as nn
import torch
from midaGAN.nn.utils import get_norm_layer_3d, is_bias_before_norm
import monai
# Config imports
from dataclasses import dataclass
from midaGAN import configs

# Network imports
from midaGAN.nn.discriminators import patchgan3d

def get_cropped_patch(input: torch.Tensor, scale: int = 1) -> torch.Tensor:
    """
    Get a downscaled patch from the input tensor.
    The scale determines how much to reduce the size by. A scale of 2 would mean a patch half the size. 

    The patch is extracted randomly from the input tensor 
    """
    # Monai transforms expect shape in CDHW format
    crop_to_shape = (input.shape[1], input.shape[2]//scale, input.shape[3]//scale, input.shape[4]//scale)
    # Random center enabled but with fixed size. 
    crop_transform = monai.transforms.RandSpatialCrop(crop_to_shape, random_center=True, random_size=False)
    cropped_input = crop_transform(input)
    return cropped_input


@dataclass
class MultiScalePatchGAN3DConfig(configs.base.BaseDiscriminatorConfig):
    name: str = "MultiScalePatchGAN3D"
    in_channels: int = 1
    ndf: int = 64
    n_layers: int = 3
    kernel_size: Tuple[int] = (4, 4, 4)

    # Each scale will reduce the input size to the discriminator by 1/x a factor. 
    # So if scales=3 the discriminator will discriminate on original, 
    # a patch 1/2 size and a patch 1/3 sized sampled randomly
    scales: int = 2


class MultiScalePatchGAN3D(nn.Module):
    def __init__(self, in_channels, ndf, n_layers, kernel_size, scales, norm_type):
        super().__init__()
        # Multiscale PatchGAN consists of multiple PatchGANs. 
        self.model = nn.ModuleDict({str(scale): patchgan3d.PatchGAN3D(in_channels, ndf, n_layers, kernel_size, norm_type) \
                                        for scale in range(1, scales + 1)})
        
    def forward(self, input):
        model_outputs = {}
        for scale, model in self.model.items():
            patch = get_cropped_patch(input, scale=int(scale))
            model_outputs[str(scale)] = model.forward(patch)

        return model_outputs




