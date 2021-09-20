from torch import nn
from ganslate.nn.utils import get_norm_layer_3d, is_bias_before_norm

# Config imports
from dataclasses import dataclass
from ganslate import configs


@dataclass
class Resnet3DConfig(configs.base.BaseGeneratorConfig):
    n_residual_blocks: int = 9


class Resnet3D(nn.Module):
    """Note: Unlike 2D version, this one uses ReplicationPad instead of RefectionPad"""

    def __init__(self, in_channels, out_channels, norm_type, n_residual_blocks=9):
        super().__init__()

        norm_layer = get_norm_layer_3d(norm_type)
        use_bias = is_bias_before_norm(norm_type)

        # Initial convolution block
        model = [
            nn.ReplicationPad3d(3),
            nn.Conv3d(in_channels, 64, 7, bias=use_bias),
            norm_layer(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv3d(in_features, out_features, 3, stride=2, padding=1, bias=use_bias),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features, norm_type)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose3d(in_features,
                                   out_features,
                                   3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReplicationPad3d(3), nn.Conv3d(64, out_channels, 7, bias=use_bias), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_features, norm_type):
        super().__init__()
        norm_layer = get_norm_layer_3d(norm_type)
        use_bias = is_bias_before_norm(norm_type)

        conv_block = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_features, in_features, 3, bias=use_bias),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_features, in_features, 3, bias=use_bias),
            norm_layer(in_features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
