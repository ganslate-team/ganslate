


class ThickBlock3d(nn.Module):
    def __init__(self, dim, use_bias, use_naive=False):
        super(ThickBlock3d, self).__init__()
        F = self.build_conv_block(dim // 2, True)
        G = self.build_conv_block(dim // 2, True)
        if use_naive:
            self.rev_block = ReversibleBlock(F, G, 'additive',
                                             keep_input=True, implementation_fwd=2, implementation_bwd=2)
        else:
            self.rev_block = ReversibleBlock(F, G, 'additive')

    def build_conv_block(self, dim, use_bias):
        conv_block = []
        conv_block += [nn.InstanceNorm3d(dim)]
        conv_block += [nn.ReplicationPad3d(1)]
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias)]
        conv_block += [nn.InstanceNorm3d(dim)]
        conv_block += [nn.ReLU(True)]
        conv_block += [nn.ReplicationPad3d(1)]
        conv_block += [ZeroInit(dim, dim, kernel_size=3, padding=0, bias=use_bias)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return self.rev_block(x)

    def inverse(self, x):
        return self.rev_block.inverse(x)       


class EdsrFGenerator3d(nn.Module):
    def __init__(self, input_nc, output_nc, depth, ngf=64, use_naive=False):
        super(EdsrFGenerator3d, self).__init__()

        use_bias = True
        downconv_ab = [nn.ReplicationPad3d(2),
                       nn.Conv3d(input_nc, ngf, kernel_size=5,
                                 stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm3d(ngf),
                       nn.ReLU(True),
                       nn.Conv3d(ngf, ngf * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       nn.InstanceNorm3d(ngf * 2),
                       nn.ReLU(True)]
        downconv_ba = [nn.ReplicationPad3d(2),
                       nn.Conv3d(input_nc, ngf, kernel_size=5,
                                 stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm3d(ngf),
                       nn.ReLU(True),
                       nn.Conv3d(ngf, ngf * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       nn.InstanceNorm3d(ngf * 2),
                       nn.ReLU(True)]

        core = []
        for _ in range(depth):
            core += [ThickBlock3d(ngf * 2, use_bias, use_naive)]

        upconv_ab = [nn.ConvTranspose3d(ngf * 2, ngf,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                       bias=use_bias),
                     nn.InstanceNorm3d(ngf),
                     nn.ReLU(True),
                     nn.ReplicationPad3d(2),
                     nn.Conv3d(ngf, output_nc, kernel_size=5, padding=0),
                     nn.Tanh()]
        upconv_ba = [nn.ConvTranspose3d(ngf * 2, ngf,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                       bias=use_bias),
                     nn.InstanceNorm3d(ngf),
                     nn.ReLU(True),
                     nn.ReplicationPad3d(2),
                     nn.Conv3d(ngf, output_nc, kernel_size=5, padding=0),
                     nn.Tanh()]

        self.downconv_ab = nn.Sequential(*downconv_ab)
        self.downconv_ba = nn.Sequential(*downconv_ba)
        self.core = nn.ModuleList(core)
        self.upconv_ab = nn.Sequential(*upconv_ab)
        self.upconv_ba = nn.Sequential(*upconv_ba)

    def forward(self, input, inverse=False):
        out = input

        if inverse:
            out = self.downconv_ba(out)
            for block in reversed(self.core):
                out = block.inverse(out)
            return self.upconv_ba(out)
        else:
            out = self.downconv_ab(out)
            for block in self.core:
                out = block(out)
            return self.upconv_ab(out)