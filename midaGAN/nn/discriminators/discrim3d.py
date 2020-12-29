import torch
import torch.nn as nn
from midaGAN.nn.utils import get_norm_layer_3d, is_bias_before_norm

# Config imports
from dataclasses import dataclass, field
from typing import Tuple, Union
from omegaconf import MISSING
from midaGAN import configs


@dataclass
class Discrim3DConfig(configs.base.BaseDiscriminatorConfig):
    name: str = "Discrim3D"
    in_channels: int = 1
    input_size: Tuple[int, int, int] = (MISSING, MISSING, MISSING)
    ndf: int = 64
    n_layers: int = 3
    kernel_size: Tuple[int] = (4, 4, 4)


class Discrim3D(nn.Module):

    def __init__(self, in_channels, input_size, ndf, n_layers, kernel_size, norm_type):
        super().__init__()

        norm_layer = get_norm_layer_3d(norm_type)
        use_bias = is_bias_before_norm(norm_type)

        kw = kernel_size
        padw = 1
        sequence = [
            nn.Conv3d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev,
                          ndf * nf_mult,
                          kernel_size=kw,
                          stride=2,
                          padding=padw,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev,
                      ndf * nf_mult,
                      kernel_size=kw,
                      stride=1,
                      padding=padw,
                      bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult,
                      in_channels,
                      kernel_size=kw,
                      stride=1,
                      padding=padw,
                      bias=use_bias)
        ]

        # ---- This part is addition, the rest is the same as in PatchGAN discriminator ----
        shape_after_convs = calc_shape_afer_convs(sequence, input_size)
        flattened_shape = torch.prod(shape_after_convs)

        sequence += [
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            Flatten(),
            nn.Linear(flattened_shape, 1)
        ]
        # ----------------------------------------------------------------------------------

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


def calc_shape_afer_convs(layer_list, input_size):
    # TODO: currently 3d only, what about pool layers
    input_size = torch.as_tensor(input_size)
    out_size = None
    for layer in layer_list:
        if isinstance(layer, nn.Conv3d):
            kernel_size = torch.as_tensor(layer.kernel_size)
            padding = torch.as_tensor(layer.padding)
            stride = torch.as_tensor(layer.stride)

            if out_size is None:
                out_size = (input_size - kernel_size + 2 * padding) // stride + 1
            else:
                out_size = (out_size - kernel_size + 2 * padding) // stride + 1
    return out_size
