import torch
import torch.nn as nn
import memcnn

class VnetRevBlock(nn.Module):
    # TODO: make an Invertible class that can wrap any block and modify it's forward func
    def __init__(self, n_channels, keep_input):
        super(VnetRevBlock, self).__init__()
        
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


class InputTransition(nn.Module):
    def __init__(self, out_channels=16):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # repeat to match channel dimension in order to perform residual connection
        x_repeated = torch.repeat(1, out_channels, 1, 1, 1) 
        out = self.relu(torch.add(out, x_repeated))
        return out


class DownTransition(nn.Module):
    def __init__(self, in_channels, n_conv_blocks, keep_input):
        super(DownTransition, self).__init__()
        out_channels = 2*in_channels
        self.down_conv_ab = self.build_down_conv(in_channels, out_channels)
        self.down_conv_ba = self.build_down_conv(in_channels, out_channels)
        # TODO: Make an Inverse sequence that does this and what's been done in line 74
        self.core = nn.Sequential(*[VnetRevBlock(out_channels, keep_input) for _ in range(n_conv_blocks)])
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


class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv_blocks, keep_input):
        super(UpTransition, self).__init__()
        self.up_conv_ab = self.build_up_conv(in_channels, out_channels)
        self.up_conv_ba = self.build_up_conv(in_channels, out_channels)

        self.core = nn.Sequential(*[VnetRevBlock(out_channels, keep_input) for _ in range(n_conv_blocks)])
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


class OutputTransition(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 2, kernel_size=5, padding=2) # should i put num_classes here as well?
        self.bn1 = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, num_classes, kernel_size=1)
        self.relu1 = nn.PReLU(num_classes)
        self.tanh = nn.Tanh() 

    def forward(self, x):
        # convolve 32 down to num of channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        res = self.tanh(out)
        return res


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, num_classes=1, keep_input=False):
        super(VNet, self).__init__()
        self.in_tr_ab = InputTransition(16)
        self.in_tr_ba = InputTransition(16)

        # ----- partially reversible layers ------

        # self.down_tr32 = DownTransition(16, 2, keep_input)
        # self.down_tr64 = DownTransition(32, 3, keep_input)
        # self.down_tr128 = DownTransition(64, 3, keep_input)
        # self.down_tr256 = DownTransition(128, 3, keep_input)

        # self.up_tr256 = UpTransition(256, 256, 3, keep_input)
        # self.up_tr128 = UpTransition(256, 128, 3, keep_input)
        # self.up_tr64 = UpTransition(128, 64, 2, keep_input)
        # self.up_tr32 = UpTransition(64, 32, 1, keep_input)

        self.down_tr32 = DownTransition(16, 1, keep_input)
        self.down_tr64 = DownTransition(32, 2, keep_input)
        self.down_tr128 = DownTransition(64, 3, keep_input)
        self.down_tr256 = DownTransition(128, 2, keep_input)

        self.up_tr256 = UpTransition(256, 256, 2, keep_input)
        self.up_tr128 = UpTransition(256, 128, 2, keep_input)
        self.up_tr64 = UpTransition(128, 64, 1, keep_input)
        self.up_tr32 = UpTransition(64, 32, 1, keep_input)
        
        # -----------------------------------------
        
        self.out_tr_ab = OutputTransition(32, num_classes)
        self.out_tr_ba = OutputTransition(32, num_classes)

    def forward(self, x, inverse=False):
        if inverse:
            in_tr  = self.in_tr_ba
            out_tr = self.out_tr_ba
        else:
            in_tr  = self.in_tr_ab
            out_tr = self.out_tr_ab
        
        out16 = in_tr(x)
        out32 = self.down_tr32(out16, inverse)
        out64 = self.down_tr64(out32, inverse)
        out128 = self.down_tr128(out64, inverse)
        out256 = self.down_tr256(out128, inverse)
        out = self.up_tr256(out256, out128, inverse)
        out = self.up_tr128(out, out64, inverse)
        out = self.up_tr64(out, out32, inverse)
        out = self.up_tr32(out, out16, inverse)
        out = out_tr(out)
        return out


class DeeperVNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, num_classes=1, keep_input=False):
        super(DeeperVNet, self).__init__()
        print('keep_input', keep_input)
        self.in_tr_ab = InputTransition(16)
        self.in_tr_ba = InputTransition(16)

        # ----- partially reversible layers ------

        self.down_tr32 = DownTransition(16, 2, keep_input)
        self.down_tr64 = DownTransition(32, 3, keep_input)
        self.down_tr128 = DownTransition(64, 3, keep_input)
        self.down_tr256 = DownTransition(128, 3, keep_input)

        self.up_tr256 = UpTransition(256, 256, 3, keep_input)
        self.up_tr128 = UpTransition(256, 128, 3, keep_input)
        self.up_tr64 = UpTransition(128, 64, 2, keep_input)
        self.up_tr32 = UpTransition(64, 32, 1, keep_input)
        
        # -----------------------------------------
        
        self.out_tr_ab = OutputTransition(32, num_classes)
        self.out_tr_ba = OutputTransition(32, num_classes)

    def forward(self, x, inverse=False):
        if inverse:
            in_tr  = self.in_tr_ba
            out_tr = self.out_tr_ba
        else:
            in_tr  = self.in_tr_ab
            out_tr = self.out_tr_ab
        
        out16 = in_tr(x)
        out32 = self.down_tr32(out16, inverse)
        out64 = self.down_tr64(out32, inverse)
        out128 = self.down_tr128(out64, inverse)
        out256 = self.down_tr256(out128, inverse)
        out = self.up_tr256(out256, out128, inverse)
        out = self.up_tr128(out, out64, inverse)
        out = self.up_tr64(out, out32, inverse)
        out = self.up_tr32(out, out16, inverse)
        out = out_tr(out)
        return out