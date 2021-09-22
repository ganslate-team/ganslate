from loguru import logger

import torch
from torch import nn

from ganslate.nn import invertible
from ganslate.nn.utils import (get_conv_layer_3d, get_conv_transpose_layer_3d, get_norm_layer_3d,
                              is_bias_before_norm)

# Config imports
from typing import Tuple
from dataclasses import dataclass
from ganslate import configs


@dataclass
class Vnet3DConfig(configs.base.BaseGeneratorConfig):
    """Partially-invertible V-Net generator."""
    use_memory_saving: bool = False  # Turn on memory saving for invertible layers. [Default: True]
    use_inverse: bool = False  # Specifies if the inverse forward will be used so that it construct the required layers
    first_layer_channels: int = 16
    down_blocks: Tuple[int] = (1, 2, 3, 2)
    up_blocks: Tuple[int] = (2, 2, 1, 1)
    is_separable: bool = False


class Vnet3D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type,
                 first_layer_channels=16,
                 down_blocks=(1, 2, 3, 2),
                 up_blocks=(2, 2, 1, 1),
                 use_memory_saving=True,
                 use_inverse=True,
                 is_separable=False):
        super().__init__()

        if use_memory_saving is False and use_inverse is False:
            disable_invertibles = True
            logger.info('Invertible layers are disabled.')
        else:
            disable_invertibles = False

        if first_layer_channels % in_channels:
            raise ValueError("`first_layer_channels` has to be divisible by `in_channels`.")
        if len(down_blocks) != len(up_blocks):
            raise ValueError("Number of `down_blocks` and `up_blocks` has to be equal.")

        keep_input = not use_memory_saving
        norm_layer = get_norm_layer_3d(norm_type)
        use_bias = is_bias_before_norm(norm_type)
        self.use_inverse = use_inverse

        # Input (first) layer
        self.in_ab = InputBlock(in_channels, first_layer_channels, norm_layer, use_bias,
                                is_separable)
        if use_inverse:
            self.in_ba = InputBlock(in_channels, first_layer_channels, norm_layer, use_bias,
                                    is_separable)

        # Output (last) layer
        self.out_ab = OutBlock(first_layer_channels * 2, out_channels, norm_layer, use_bias,
                               is_separable)
        if use_inverse:
            self.out_ba = OutBlock(first_layer_channels * 2, out_channels, norm_layer, use_bias,
                                   is_separable)

        # Downblocks
        downs = []
        down_channel_factors = []
        for i, num_convs in enumerate(down_blocks):
            factor = 2**i  # gives the 1, 2, 4, 8, 16, etc. series
            downs += [
                DownBlock(first_layer_channels * factor, num_convs, norm_layer, use_bias,
                          keep_input, use_inverse, disable_invertibles, is_separable)
            ]
            down_channel_factors.append(factor)
        self.downs = nn.ModuleList(downs)

        # NOTE: in order to be able to use an architecture for CUT, it is necessary to
        # have self.encoder which contains all the layers of the encoder part of the network
        # and that is iterable chronologically. It's a PyTorch limitation as .children()
        # on a network won't yield the layers in the order in which they are called in the forward
        # pass, but rather in the order in which they were initialized.
        self.encoder = nn.ModuleList([self.in_ab]).extend(self.downs)

        # Upblocks
        up_channel_factors = [factor * 2 for factor in reversed(down_channel_factors)]
        num_convs = up_blocks[0]
        ups = [
            UpBlock(first_layer_channels * up_channel_factors[0],
                    first_layer_channels * up_channel_factors[0], num_convs, norm_layer, use_bias,
                    keep_input, use_inverse, disable_invertibles, is_separable)
        ]

        for i, num_convs in enumerate(up_blocks[1:]):
            ups += [
                UpBlock(first_layer_channels * up_channel_factors[i],
                        first_layer_channels * up_channel_factors[i + 1], num_convs, norm_layer,
                        use_bias, keep_input, use_inverse, disable_invertibles, is_separable)
            ]
        self.ups = nn.ModuleList(ups)

    def forward(self, x, inverse=False):
        if inverse:
            if not self.use_inverse:
                raise ValueError(
                    "Trying to perform inverse forward while `use_inverse` flag is turned off.")
            in_block = self.in_ba
            out_block = self.out_ba
        else:
            in_block = self.in_ab
            out_block = self.out_ab

        # Input block, it's output is used as last skip connection
        out1 = in_block(x)

        # Downblocks
        down_outs = []
        for i, down in enumerate(self.downs):
            if i == 0:
                down_outs += [down(out1, inverse)]
            else:
                down_outs += [down(down_outs[-1], inverse)]

        # Upblocks
        # reverse the order of outputs of downblocks to fetch skip connections chronologically
        down_outs_reversed = list(reversed(down_outs))
        for i, up in enumerate(self.ups):
            # the input of the first upblock is the output of the last downblock
            if i == 0:
                out = down_outs_reversed[i]

            # if last up_block, skip connection is the output of the input block
            if i == len(self.ups) - 1:
                skip = out1
            # otherwise it is the output of the downblock from its appropriate level
            else:
                skip = down_outs_reversed[i + 1]

            out = up(out, skip, inverse)

        # Out block
        out = out_block(out)
        return out


class InputBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, use_bias, is_separable):
        super().__init__()
        self.n_repeats = out_channels // in_channels  # how many times an image has to be repeated to match `out_channels`

        conv_layer = get_conv_layer_3d(is_separable)
        self.conv1 = conv_layer(in_channels, out_channels, kernel_size=5, padding=2, bias=use_bias)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        x_repeated = x.repeat(1, self.n_repeats, 1, 1,
                              1)  # match channel dimension for residual connection
        out = out + x_repeated
        return self.relu(out)


class DownBlock(nn.Module):

    def __init__(self, in_channels, n_conv_blocks, norm_layer, use_bias, keep_input, use_inverse,
                 disable_invertibles, is_separable):
        super().__init__()
        self.is_separable = is_separable

        out_channels = 2 * in_channels
        self.down_conv_ab = self.build_down_conv(in_channels, out_channels, norm_layer, use_bias)
        if use_inverse:
            self.down_conv_ba = self.build_down_conv(in_channels, out_channels, norm_layer,
                                                     use_bias)

        inv_block = _base_inv_block(out_channels, norm_layer, use_bias, is_separable)
        self.core = invertible.InvertibleSequence(inv_block, n_conv_blocks, keep_input,
                                                  disable_invertibles)
        self.relu = nn.PReLU(out_channels)

    def build_down_conv(self, in_channels, out_channels, norm_layer, use_bias):
        conv_layer = get_conv_layer_3d(self.is_separable)
        return nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=2, stride=2, bias=use_bias),
            norm_layer(out_channels), nn.PReLU(out_channels))

    def forward(self, x, inverse=False):
        if inverse:
            down_conv = self.down_conv_ba
        else:
            down_conv = self.down_conv_ab
        down = down_conv(x)
        out = self.core(down, inverse)
        out = out + down
        return self.relu(out)


class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_conv_blocks, norm_layer, use_bias, keep_input,
                 use_inverse, disable_invertibles, is_separable):
        super().__init__()
        self.is_separable = is_separable

        self.up_conv_ab = self.build_up_conv(in_channels, out_channels, norm_layer, use_bias)
        if use_inverse:
            self.up_conv_ba = self.build_up_conv(in_channels, out_channels, norm_layer, use_bias)

        inv_block = _base_inv_block(out_channels, norm_layer, use_bias, is_separable)
        self.core = invertible.InvertibleSequence(inv_block, n_conv_blocks, keep_input,
                                                  disable_invertibles)
        self.relu = nn.PReLU(out_channels)

    def build_up_conv(self, in_channels, out_channels, norm_layer, use_bias):
        conv_transp_layer = get_conv_transpose_layer_3d(self.is_separable)
        return nn.Sequential(
            conv_transp_layer(in_channels,
                              out_channels // 2,
                              kernel_size=2,
                              stride=2,
                              bias=use_bias), norm_layer(out_channels // 2),
            nn.PReLU(out_channels // 2))

    def forward(self, x, skipx, inverse=False):
        if inverse:
            up_conv = self.up_conv_ba
        else:
            up_conv = self.up_conv_ab
        up = up_conv(x)
        xcat = torch.cat((up, skipx), 1)
        out = self.core(xcat, inverse)
        out = out + xcat
        return self.relu(out)


class OutBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, use_bias, is_separable):
        super().__init__()
        conv_layer = get_conv_layer_3d(is_separable)

        self.conv1 = conv_layer(in_channels, in_channels, kernel_size=5, padding=2, bias=use_bias)
        self.bn1 = norm_layer(in_channels)
        self.relu1 = nn.PReLU(in_channels)
        self.conv2 = conv_layer(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        res = self.tanh(out)
        return res


def _base_inv_block(n_channels, norm_layer, use_bias, is_separable):
    n_channels = n_channels // 2  # split across channels for invertible module
    conv_layer = get_conv_layer_3d(is_separable)
    return nn.Sequential(
        conv_layer(n_channels, n_channels, kernel_size=5, padding=2, bias=use_bias),
        norm_layer(n_channels), nn.PReLU(n_channels))
