import torch
import torch.nn as nn
from models.util import get_norm_layer, is_bias_before_norm

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, n_channels_input, start_n_filters_D, n_layers_D, norm_layer_type, use_sigmoid, **_kwargs):
        super(NLayerDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_layer_type)
        use_bias = is_bias_before_norm(norm_layer_type)

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(n_channels_input, start_n_filters_D, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers_D):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(start_n_filters_D * nf_mult_prev, start_n_filters_D * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(start_n_filters_D * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers_D, 8)
        sequence += [
            nn.Conv3d(start_n_filters_D * nf_mult_prev, start_n_filters_D * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(start_n_filters_D * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(start_n_filters_D * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
