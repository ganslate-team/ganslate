
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler
import memcnn
from torch.nn import Parameter
import numpy as np
import re

# ---------------- Rev V-NET -------------------

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
            super()._check_input_dim(input) # added 4 spaces to indent this line into "if". not sure if right by Chao??
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class RevBlock(nn.Module):
    def __init__(self, nchan, elu):
        super(RevBlock, self).__init__()
        
        invertible_module = memcnn.AdditiveCoupling(
            Fm=self.build_conv_block(nchan//2, elu),
            Gm=self.build_conv_block(nchan//2, elu)
        )

        self.rev_block = memcnn.InvertibleModuleWrapper(fn=invertible_module, 
                                                        keep_input=True, 
                                                        keep_input_inverse=True)

    def build_conv_block(self, nchan, elu):
        block = nn.Sequential(nn.Conv3d(nchan, nchan, kernel_size=5, padding=2),
                              ContBatchNorm3d(nchan),
                              ELUCons(elu, nchan))
        return block

    def forward(self, x):
        residual = x
        out = self.rev_block(x)
        out = torch.add(out, residual) 
        return out

    def inverse(self, x):
        residual = x
        out = self.rev_block.inverse(x)
        out = torch.add(out, residual) 
        return out


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        # print("x before InputTransition shape:"+str(x.shape))
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels (or is it right to say duplicate the input for 16 times to operate "torch.add()"?? By Chao) 
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1) # changed dim = 0 to 1 to operate on channels, and have "x16" the same size as "out" to operate "torch.add()". By Chao.
        # print("x16 shape:"+str(x16.shape))
        # pdb.set_trace()
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv_ab = self.build_down_conv(inChans, outChans, elu)
        self.down_conv_ba = self.build_down_conv(inChans, outChans, elu)

        self.core = nn.Sequential(*[RevBlock(outChans, elu) for _ in range(nConvs)])
        self.relu = ELUCons(elu, outChans)

    def build_down_conv(self, inChans, outChans, elu):
        return nn.Sequential(nn.Conv3d(inChans, outChans, kernel_size=2, stride=2),
                             ContBatchNorm3d(outChans),
                             ELUCons(elu, outChans))

    def forward(self, x, inverse=False):
        if inverse:
            down = self.down_conv_ba(x)
            out = down.clone()
            for block in reversed(self.core):
                out = block.inverse(out)
        else:
            down = self.down_conv_ab(x)
            out = down.clone()
            for block in self.core:
                out = block(out)
        return self.relu(torch.add(out, down))


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu):
        super(UpTransition, self).__init__()
        self.up_conv_ab = self.build_up_conv(inChans, outChans, elu)
        self.up_conv_ba = self.build_up_conv(inChans, outChans, elu)

        self.core = nn.Sequential(*[RevBlock(outChans, elu) for _ in range(nConvs)])
        self.relu = ELUCons(elu, outChans)
    
    def build_up_conv(self, inChans, outChans, elu):
        return nn.Sequential(nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2),
                             ContBatchNorm3d(outChans // 2),
                             ELUCons(elu, outChans // 2))

    def forward(self, x, skipx, inverse=False):
        if inverse:
            out = self.up_conv_ba(x)
            xcat = torch.cat((out, skipx), 1)
            out = xcat.clone()
            for block in reversed(self.core):
                out = block.inverse(out)
        else:
            out = self.up_conv_ab(x)
            xcat = torch.cat((out, skipx), 1)
            out = xcat.clone()
            for block in self.core:
                out = block(out)

        return self.relu(torch.add(out, xcat))


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll, num_classes):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2) # should i put num_classes here as well?
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, num_classes, kernel_size=1)
        self.relu1 = ELUCons(elu, num_classes)
        # TAKE CARE OF THIS AS A GEN IT SHOULD BE TANH
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = nn.Tanh() #F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # print("out shape before softmax:"+str(out.shape))

        # flatten z, y, x
        #b, c, z, y, x = out.shape # b:batch_size, c:channels, z:depth, y:height, w:width. channels is 2? as the output channels of the last conv layer?
        #out = out.view(b, c, -1)

        # pdb.set_trace()
        res = self.softmax(out)#, dim = 1)

        # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out,dim=1)
        return res


class RevResVNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False, num_classes=1):
        super(RevResVNet, self).__init__()

        self.in_tr_ab = InputTransition(16, elu)
        self.in_tr_ba = InputTransition(16, elu)

        # ----- partially reversible layers ------
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu)
        self.down_tr256 = DownTransition(128, 2, elu)

        self.up_tr256 = UpTransition(256, 256, 2, elu)
        self.up_tr128 = UpTransition(256, 128, 2, elu)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        # -----------------------------------------
        
        self.out_tr_ab = OutputTransition(32, elu, nll, num_classes)
        self.out_tr_ba = OutputTransition(32, elu, nll, num_classes)

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