import torch
import torch.nn as nn
import memcnn

class VnetRevBlock(nn.Module):
    # TODO: make an Invertible class that can wrap any block and modify it's forward func
    def __init__(self, nchan, keep_input):
        super(VnetRevBlock, self).__init__()
        
        invertible_module = memcnn.AdditiveCoupling(
            Fm=self.build_conv_block(nchan//2),
            Gm=self.build_conv_block(nchan//2)
        )
        self.rev_block = memcnn.InvertibleModuleWrapper(fn=invertible_module, 
                                                        keep_input=keep_input, 
                                                        keep_input_inverse=keep_input)

    def build_conv_block(self, nchan):
        block = nn.Sequential(nn.Conv3d(nchan, nchan, kernel_size=5, padding=2),
                              nn.BatchNorm3d(nchan),
                              nn.PReLU(nchan))
        return block

    def forward(self, x, inverse=False):
        if inverse:
            return self.rev_block.inverse(x)
        else:
            return self.rev_block(x)


class InputTransition(nn.Module):
    def __init__(self, outChans):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.PReLU(16)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels (or is it right to say duplicate the input for 16 times to operate "torch.add()"?? By Chao) 
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1) # changed dim = 0 to 1 to operate on channels, and have "x16" the same size as "out" to operate "torch.add()". By Chao.
        out = self.relu(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, keep_input):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv_ab = self.build_down_conv(inChans, outChans)
        self.down_conv_ba = self.build_down_conv(inChans, outChans)
        # TODO: Make an Inverse sequence that does this and what's been done in line 74
        self.core = nn.Sequential(*[VnetRevBlock(outChans, keep_input) for _ in range(nConvs)])
        self.relu = nn.PReLU(outChans)

    def build_down_conv(self, inChans, outChans):
        return nn.Sequential(nn.Conv3d(inChans, outChans, kernel_size=2, stride=2),
                             nn.BatchNorm3d(outChans),
                             nn.PReLU(outChans))

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
    def __init__(self, inChans, outChans, nConvs, keep_input):
        super(UpTransition, self).__init__()
        self.up_conv_ab = self.build_up_conv(inChans, outChans)
        self.up_conv_ba = self.build_up_conv(inChans, outChans)

        self.core = nn.Sequential(*[VnetRevBlock(outChans, keep_input) for _ in range(nConvs)])
        self.relu = nn.PReLU(outChans)
    
    def build_up_conv(self, inChans, outChans):
        return nn.Sequential(nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2),
                             nn.BatchNorm3d(outChans // 2),
                             nn.PReLU(outChans // 2))

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
    def __init__(self, inChans, num_classes):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2) # should i put num_classes here as well?
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