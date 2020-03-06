import torch
from torch import nn, optim
import memcnn


class RevBlock(nn.Module):
    def __init__(self, nchan):
        super(RevBlock, self).__init__()

        invertible_module = memcnn.AdditiveCoupling(
            Fm=self.build_conv_block(nchan//2),
            Gm=self.build_conv_block(nchan//2)
        )

        self.rev_block = memcnn.InvertibleModuleWrapper(fn=invertible_module, 
                                                        keep_input=False, 
                                                        keep_input_inverse=False)

    def build_conv_block(self, nchan):
        return nn.Sequential(nn.Conv3d(nchan, nchan, kernel_size=5, padding=2),
                             nn.BatchNorm3d(nchan),
                             nn.PReLU(nchan))
        
    def forward(self, x, inverse=False):
        if inverse:
            return self.rev_block.inverse(x)
        else:
            return self.rev_block(x)


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv_ab = self.build_down_conv(inChans, outChans)
        self.down_conv_ba = self.build_down_conv(inChans, outChans)
        self.core = nn.Sequential(*[RevBlock(outChans) for _ in range(nConvs)])
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
        out = down    # the reason it breaks
        for block in core:
            out = block(out, inverse=inverse)
        
        #out = out + down
        return self.relu(out)


model = DownTransition(16, 2)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for i in range(10):
    optimizer.zero_grad()
    data, target = torch.rand((2,16,64,64,64)), torch.rand((2,32,32,32,32))
    #data.requires_grad = True
    out = model.forward(data)
    out = model.forward(data, inverse=True)
    print('both done')
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()