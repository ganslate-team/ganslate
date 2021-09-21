from loguru import logger
from dataclasses import dataclass
# Config imports
from typing import Tuple

import torch
from torch import nn
from ganslate import configs
from ganslate.nn import invertible
from ganslate.nn import attention

from ganslate.nn.generators.vnet.vnet3d import (DownBlock, InputBlock, OutBlock, UpBlock)
from ganslate.nn.utils import (get_conv_layer_3d, get_conv_transpose_layer_3d, get_norm_layer_3d,
                              is_bias_before_norm)


@dataclass
class SelfAttentionVnet3DConfig(configs.base.BaseGeneratorConfig):
    """Partially-invertible V-Net generator with Self-Attention

    Self Attention Blocks are added at each DownBlock. 
    SA original publication: https://arxiv.org/pdf/1805.08318.pdf
    Idea: Regions in image can escape locality constraint in CNNs by 
    forming query and key pairs and comparing generating an attention map
    based on learnable relations between these pairs. 
    Usecase: Might be good for CBCT -> CT so that artifacts in different
    regions can be associated as something that needs to be rectified for CT
    translation
    
    """
    use_memory_saving: bool = True  # Turn on memory saving for invertible layers. [Default: True]
    use_inverse: bool = True  # Specifies if the inverse forward will be used so that it construct the required layers
    first_layer_channels: int = 16
    down_blocks: Tuple[int] = (1, 2, 3, 2)
    up_blocks: Tuple[int] = (2, 2, 1, 1)
    is_separable: bool = False

    # Need to correspond to the same length as the number of down blocks
    enable_attention_block: Tuple[bool] = (False, False, True, True)


class SelfAttentionVnet3D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type,
                 first_layer_channels=16,
                 down_blocks=(1, 2, 3, 2),
                 up_blocks=(2, 2, 1, 1),
                 use_memory_saving=True,
                 use_inverse=True,
                 enable_attention_block=(True, True, True, True),
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
        attention_blocks = []
        down_channel_factors = []
        for i, num_convs in enumerate(down_blocks):
            factor = 2**i  # gives the 1, 2, 4, 8, 16, etc. series
            downs += [
                DownBlock(first_layer_channels * factor, num_convs, norm_layer, use_bias,
                          keep_input, use_inverse, disable_invertibles, is_separable)
            ]

            if enable_attention_block[i]:
                attn_block = attention.SelfAttentionBlock(first_layer_channels * factor * 2, 'relu')

            else:
                attn_block = nn.Identity()

            attention_blocks += [attn_block]

            down_channel_factors.append(factor)

        self.downs = nn.ModuleList(downs)
        self.attn_blocks = nn.ModuleList(attention_blocks)

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
        for i, (down, attn) in enumerate(zip(self.downs, self.attn_blocks)):
            if i == 0:
                down_out = down(out1, inverse)
                # Attention goes here
                down_outs += [attn(down_out)]
            else:
                down_out = down(down_outs[-1], inverse)
                # Attention goes here
                down_outs += [attn(down_out)]

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
