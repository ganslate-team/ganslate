import torch
import torch.nn as nn
import memcnn
from models.util import get_norm_layer, is_bias_before_norm


class VNet(nn.Module):
    def __init__(self, start_n_filters, norm_type, use_memory_saving):
        super(VNet, self).__init__()
        keep_input = not use_memory_saving
        norm_layer = get_norm_layer(norm_type)
        use_bias = is_bias_before_norm(norm_type)

        self.in_ab = InputBlock(start_n_filters) 
        self.in_ba = InputBlock(start_n_filters)

        self.down1 = DownBlock(start_n_filters, 1, keep_input)
        self.down2 = DownBlock(start_n_filters*2, 2, keep_input) 
        self.down3 = DownBlock(start_n_filters*4, 3, keep_input) 
        self.down4 = DownBlock(start_n_filters*8, 2, keep_input) 

        self.up4 = UpBlock(start_n_filters*16, start_n_filters*16, 2, keep_input) 
        self.up3 = UpBlock(start_n_filters*16, start_n_filters*8, 2, keep_input) 
        self.up2 = UpBlock(start_n_filters*8, start_n_filters*4, 1, keep_input) 
        self.up1 = UpBlock(start_n_filters*4, start_n_filters*2, 1, keep_input) 
        
        self.out_ab = OutBlock(start_n_filters*2) 
        self.out_ba = OutBlock(start_n_filters*2)

    def forward(self, x, inverse=False):
        if inverse:
            in_block  = self.in_ba
            out_block = self.out_ba
        else:
            in_block  = self.in_ab
            out_block = self.out_ab
        
        out1 = in_block(x)
        out2 = self.down1(out1, inverse)
        out3 = self.down2(out2, inverse)
        out4 = self.down3(out3, inverse)
        out = self.down4(out4, inverse)
        out = self.up4(out, out4, inverse)
        out = self.up3(out, out3, inverse)
        out = self.up2(out, out2, inverse)
        out = self.up1(out, out1, inverse)
        out = out_block(out)
        return out

class InvertibleBlock(nn.Module):
    # TODO: make an Invertible class that can wrap any block and modify it's forward func
    def __init__(self, n_channels, keep_input):
        super(InvertibleBlock, self).__init__()
        
        invertible_module = memcnn.AdditiveCoupling(
            Fm=self.build_conv_block(n_channels//2),
            Gm=self.build_conv_block(n_channels//2)
        )
        self.rev_block = memcnn.InvertibleModuleWrapper(fn=invertible_module, 
                                                        keep_input=keep_input, 
                                                        keep_input_inverse=keep_input)

    def build_conv_block(self, n_channels):
        block = nn.Sequential(nn.Conv3d(n_channels, n_channels, kernel_size=5, padding=2),
                              nn.BatchNorm3d(n_channels),
                              nn.PReLU(n_channels))
        return block

    def forward(self, x, inverse=False):
        if inverse:
            return self.rev_block.inverse(x)
        else:
            return self.rev_block(x)


class InputBlock(nn.Module):
    def __init__(self, out_channels):
        super(InputBlock, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv3d(1, out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        x_repeated = x.repeat(1, self.out_channels, 1, 1, 1) # match channel dimension for residual connection
        out = self.relu(torch.add(out, x_repeated))
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, n_conv_blocks, keep_input):
        super(DownBlock, self).__init__()
        out_channels = 2*in_channels
        self.down_conv_ab = self.build_down_conv(in_channels, out_channels)
        self.down_conv_ba = self.build_down_conv(in_channels, out_channels)
        # TODO: Make an Inverse sequence that does this
        self.core = nn.Sequential(*[InvertibleBlock(out_channels, keep_input) for _ in range(n_conv_blocks)])
        self.relu = nn.PReLU(out_channels)

    def build_down_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
                             nn.BatchNorm3d(out_channels),
                             nn.PReLU(out_channels))

    def forward(self, x, inverse=False):
        if inverse:
            down_conv = self.down_conv_ba
            core = reversed(self.core)
        else:
            down_conv = self.down_conv_ab
            core = self.core

        down = down_conv(x)
        out = down
        # TODO: Make an Inverse sequence that does this
        for i, block in enumerate(core):
            if i == 0:
                #https://github.com/silvandeleemput/memcnn/issues/39#issuecomment-599199122
                if inverse:
                    block.rev_block.keep_input_inverse = True
                else:
                    block.rev_block.keep_input = True
            out = block(out, inverse=inverse)
        
        out = out + down
        return self.relu(out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv_blocks, keep_input):
        super(UpBlock, self).__init__()
        self.up_conv_ab = self.build_up_conv(in_channels, out_channels)
        self.up_conv_ba = self.build_up_conv(in_channels, out_channels)

        self.core = nn.Sequential(*[InvertibleBlock(out_channels, keep_input) for _ in range(n_conv_blocks)])
        self.relu = nn.PReLU(out_channels)
    
    def build_up_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2),
                             nn.BatchNorm3d(out_channels // 2),
                             nn.PReLU(out_channels // 2))

    def forward(self, x, skipx, inverse=False):
        if inverse:
            up_conv = self.up_conv_ba
            core = reversed(self.core)
        else:
            up_conv = self.up_conv_ab
            core = self.core

        up = up_conv(x)
        xcat = torch.cat((up, skipx), 1)
        out = xcat
        for i, block in enumerate(core):
            if i == 0:
                #https://github.com/silvandeleemput/memcnn/issues/39#issuecomment-599199122
                if inverse:
                    block.rev_block.keep_input_inverse = True
                else:
                    block.rev_block.keep_input = True
            out = block(out, inverse)

        out = out + xcat
        return self.relu(out)


class OutBlock(nn.Module):
    def __init__(self, in_channels):
        super(OutBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.relu1 = nn.PReLU(1)
        self.tanh = nn.Tanh() 

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        res = self.tanh(out)
        return res

