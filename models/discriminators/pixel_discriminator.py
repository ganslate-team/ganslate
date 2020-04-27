import torch
import torch.nn as nn
import functools

class PixelDiscriminator(nn.Module):
    def __init__(self, n_channels_input, start_n_filters_D=64, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

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