from torch import nn
from torch.nn.modules.utils import _triple


class SeparableConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        depthwise_kernel = (1, kernel_size[1], kernel_size[2])
        pointwise_kernel = (kernel_size[0], 1, 1)

        depthwise_stride = (1, stride[1], stride[2])
        pointwise_stride = (stride[0], 1, 1)

        depthwise_padding = (0, padding[1], padding[2])
        pointwise_padding = (padding[0], 0, 0)

        # construct the layers
        self.conv_depthwise = nn.Conv3d(in_channels,
                                        out_channels,
                                        kernel_size=depthwise_kernel,
                                        stride=depthwise_stride,
                                        padding=depthwise_padding,
                                        bias=bias)

        self.conv_pointwise = nn.Conv3d(out_channels,
                                        out_channels,
                                        kernel_size=pointwise_kernel,
                                        stride=pointwise_stride,
                                        padding=pointwise_padding,
                                        bias=bias)

    def forward(self, input):
        out = self.conv_depthwise(input)
        out = self.conv_pointwise(out)
        return out


class SeparableConvTranspose3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        depthwise_kernel = (1, kernel_size[1], kernel_size[2])
        pointwise_kernel = (kernel_size[0], 1, 1)

        depthwise_stride = (1, stride[1], stride[2])
        pointwise_stride = (stride[0], 1, 1)

        depthwise_padding = (0, padding[1], padding[2])
        pointwise_padding = (padding[0], 0, 0)

        # construct the layers
        self.conv_transp_depthwise = nn.ConvTranspose3d(in_channels,
                                                        out_channels,
                                                        kernel_size=depthwise_kernel,
                                                        stride=depthwise_stride,
                                                        padding=depthwise_padding,
                                                        bias=bias)

        self.conv_transp_pointwise = nn.ConvTranspose3d(out_channels,
                                                        out_channels,
                                                        kernel_size=pointwise_kernel,
                                                        stride=pointwise_stride,
                                                        padding=pointwise_padding,
                                                        bias=bias)

    def forward(self, input):
        out = self.conv_transp_depthwise(input)
        out = self.conv_transp_pointwise(out)
        return out
