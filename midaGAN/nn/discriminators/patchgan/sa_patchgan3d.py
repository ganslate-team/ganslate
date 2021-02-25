from typing import Tuple
import torch.nn as nn
from midaGAN.nn.utils import get_norm_layer_3d, is_bias_before_norm
from midaGAN.nn import attention

# Config imports
from dataclasses import dataclass
from midaGAN import configs


@dataclass
class SAPatchGAN3DConfig(configs.base.BaseDiscriminatorConfig):
    name: str = "SAPatchGAN3D"
    in_channels: int = 1
    ndf: int = 64
    n_layers: int = 3
    kernel_size: Tuple[int] = (4, 4, 4)


class SAPatchGAN3D(nn.Module):

    def __init__(self, in_channels, ndf, n_layers, kernel_size, norm_type):
        super().__init__()

        norm_layer = get_norm_layer_3d(norm_type)
        use_bias = is_bias_before_norm(norm_type)

        kw = kernel_size
        padw = 1
        sequence = [
            nn.Conv3d(in_channels, ndf, kernel_size=kw, stride=3, padding=padw),
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
        sequence += [attention.SelfAttentionBlock(ndf * nf_mult, 'relu')]
        sequence += [nn.Conv3d(ndf * nf_mult, in_channels, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)