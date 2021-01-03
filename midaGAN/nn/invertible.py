from copy import deepcopy

from torch import nn

import memcnn


class InvertibleBlock(nn.Module):

    def __init__(self, block, keep_input, disable=False):
        """The input block should already be split across channels # TODO: explain better
        """
        super().__init__()

        block = memcnn.AdditiveCoupling(deepcopy(block))
        self.invertible_block = memcnn.InvertibleModuleWrapper(fn=block,
                                                               keep_input=keep_input,
                                                               keep_input_inverse=keep_input,
                                                               disable=disable)

    def forward(self, x, inverse=False):
        if inverse:
            return self.invertible_block.inverse(x)
        return self.invertible_block(x)


class InvertibleSequence(nn.Module):

    def __init__(self, block, n_blocks, keep_input, disable=False):
        super().__init__()

        sequence = [InvertibleBlock(block, keep_input, disable) for _ in range(n_blocks)]
        self.sequence = nn.Sequential(*sequence)

    def forward(self, x, inverse=False):
        if inverse:
            sequence = reversed(self.sequence)
        else:
            sequence = self.sequence

        for i, block in enumerate(sequence):
            if i == 0:  #https://github.com/silvandeleemput/memcnn/issues/39#issuecomment-599199122
                if inverse:
                    block.invertible_block.keep_input_inverse = True
                else:
                    block.invertible_block.keep_input = True
            x = block(x, inverse=inverse)
        return x
