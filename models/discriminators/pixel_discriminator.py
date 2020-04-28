import torch
import torch.nn as nn
from models.util import get_norm_layer, is_bias_before_norm


class PixelDiscriminator(nn.Module):
    def __init__(self, n_channels_input, start_n_filters_D, norm_layer_type, use_sigmoid, **_kwargs):
        super(PixelDiscriminator, self).__init__()
        
        norm_layer = get_norm_layer(norm_layer_type)
        use_bias = is_bias_before_norm(norm_layer_type)

        self.net = [
            nn.Conv3d(n_channels_input, start_n_filters_D, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(start_n_filters_D, start_n_filters_D * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(start_n_filters_D * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(start_n_filters_D * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)