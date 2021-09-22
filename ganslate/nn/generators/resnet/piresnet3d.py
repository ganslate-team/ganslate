# Taken from RevGAN (https://github.com/tychovdo/RevGAN/blob/2af25e6a8176eaab3d424db45fb6ee2cfc5dc9a3/models/networks3d.py#L380)
# Originally names as EdsrFGenerator3d
# Changes:
#   - port to the latest memcnn
#   - refactor
#   - change name to Piresnet (partially-invertible Resnet) as I can't find what EdsrF stands for

from torch import nn

from ganslate.nn import invertible
from ganslate.nn.utils import get_norm_layer_3d, is_bias_before_norm

# Config imports
from dataclasses import dataclass
from omegaconf import MISSING
from ganslate import configs


@dataclass
class Piresnet3DConfig(configs.base.BaseGeneratorConfig):
    """Partially-invertible Resnet generator - a version of ResNet compatible with RevGAN."""
    use_memory_saving: bool = True  # Turn on memory saving for invertible layers. [Default: True]
    use_inverse: bool = True  # Specifies if the inverse forward will be used so that it construct the required layers
    first_layer_channels: int = 32
    depth: int = MISSING


class Piresnet3D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type,
                 depth,
                 first_layer_channels=64,
                 use_memory_saving=True,
                 use_inverse=True):
        super().__init__()
        # TODO: implement disable_invertibles as in Vnet

        keep_input = not use_memory_saving
        norm_layer = get_norm_layer_3d(norm_type)
        use_bias = is_bias_before_norm(norm_type)
        self.use_inverse = use_inverse

        self.downconv_ab = self.build_downconv(in_channels, norm_layer, first_layer_channels,
                                               use_bias)
        self.upconv_ab = self.build_upconv(out_channels, norm_layer, first_layer_channels, use_bias)
        if use_inverse:
            self.downconv_ba = self.build_downconv(in_channels, norm_layer, first_layer_channels,
                                                   use_bias)
            self.upconv_ba = self.build_upconv(out_channels, norm_layer, first_layer_channels,
                                               use_bias)

        inv_block = _base_inv_block(first_layer_channels * 2, norm_layer, use_bias)
        self.core = invertible.InvertibleSequence(inv_block, depth, keep_input)

    def build_downconv(self, in_channels, norm_layer, first_layer_channels, use_bias):
        return nn.Sequential(
            nn.ReplicationPad3d(2),
            nn.Conv3d(in_channels,
                      first_layer_channels,
                      kernel_size=5,
                      stride=1,
                      padding=0,
                      bias=use_bias), norm_layer(first_layer_channels), nn.ReLU(inplace=True),
            nn.Conv3d(first_layer_channels,
                      first_layer_channels * 2,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=use_bias), norm_layer(first_layer_channels * 2), nn.ReLU(inplace=True))

    def build_upconv(self, out_channels, norm_layer, first_layer_channels, use_bias):
        return nn.Sequential(
            nn.ConvTranspose3d(first_layer_channels * 2,
                               first_layer_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=use_bias), norm_layer(first_layer_channels),
            nn.ReLU(inplace=True), nn.ReplicationPad3d(2),
            nn.Conv3d(first_layer_channels, out_channels, kernel_size=5, padding=0), nn.Tanh())

    def forward(self, x, inverse=False):
        out = x

        if inverse:
            if not self.use_inverse:
                raise ValueError(
                    "Trying to perform inverse forward while `use_inverse` flag is turned off.")
            downconv = self.downconv_ba
            upconv = self.upconv_ba
        else:
            downconv = self.downconv_ab
            upconv = self.upconv_ab

        out = downconv(out)
        out = self.core(out, inverse)
        return upconv(out)


def _base_inv_block(n_channels, norm_layer, use_bias):
    n_channels = n_channels // 2  # split across channels for invertible module
    return nn.Sequential(norm_layer(n_channels), nn.ReplicationPad3d(1),
                         nn.Conv3d(n_channels, n_channels, kernel_size=3, padding=0, bias=use_bias),
                         norm_layer(n_channels), nn.ReLU(inplace=True))
