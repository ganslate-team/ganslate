# Source: https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
# Changes made:
# Change 2D Conv to 3D Conv
# N (dimension of query, key and, value vectors) is DxWxH instead of WxH
# Removed attention matrix return in the forward pass

import torch
from torch import nn


# Both discriminator and generator can use attention blocks
class SelfAttentionBlock(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X D X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Depth*Width*Height)
        """
        m_batchsize, C, depth, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1,
                                             depth * width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, depth * width * height)  # B X C x (D*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, depth * width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, depth, width, height)

        out = self.gamma * out + x
        return out
