import torch
from torch import nn
from ganslate.nn.utils import get_norm_layer_2d, is_bias_before_norm

# Config imports
from dataclasses import dataclass
from ganslate import configs


@dataclass
class Unet2DConfig(configs.base.BaseGeneratorConfig):
    num_downs: int = 7
    ngf: int = 64
    use_dropout: bool = False


class Unet2D(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, in_channels, out_channels, num_downs, norm_type, ngf=64, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            in_channels (int)  -- the number of channels in input images
            out_channels (int) -- the number of channels in model's output
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_type      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8,
                                             ngf * 8,
                                             in_channels=None,
                                             submodule=None,
                                             norm_type=norm_type,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8,
                                                 ngf * 8,
                                                 in_channels=None,
                                                 submodule=unet_block,
                                                 norm_type=norm_type,
                                                 use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4,
                                             ngf * 8,
                                             in_channels=None,
                                             submodule=unet_block,
                                             norm_type=norm_type)
        unet_block = UnetSkipConnectionBlock(ngf * 2,
                                             ngf * 4,
                                             in_channels=None,
                                             submodule=unet_block,
                                             norm_type=norm_type)
        unet_block = UnetSkipConnectionBlock(ngf,
                                             ngf * 2,
                                             in_channels=None,
                                             submodule=unet_block,
                                             norm_type=norm_type)
        self.model = UnetSkipConnectionBlock(out_channels,
                                             ngf,
                                             in_channels=in_channels,
                                             submodule=unet_block,
                                             outermost=True,
                                             norm_type=norm_type)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 norm_type,
                 in_channels=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            in_channels (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_type          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super().__init__()
        self.outermost = outermost

        norm_layer = get_norm_layer_2d(norm_type)
        use_bias = is_bias_before_norm(norm_type)

        if in_channels is None:
            in_channels = outer_nc
        downconv = nn.Conv2d(in_channels,
                             inner_nc,
                             kernel_size=4,
                             stride=2,
                             padding=1,
                             bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc,
                                        outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2,
                                        outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)
