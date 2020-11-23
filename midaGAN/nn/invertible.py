import torch 
from torch import nn
from torch.cuda.amp import autocast

import memcnn


class InvertibleBlock(nn.Module):
    def __init__(self, block, keep_input):
        """The input block should already be split across channels # TODO: explain better
        """
        super().__init__()

        invertible_module = memcnn.AdditiveCoupling(block)
        self.invertible_block = memcnn.InvertibleModuleWrapper(fn=invertible_module, 
                                                               keep_input=keep_input, 
                                                               keep_input_inverse=keep_input)

    
    def forward(self, x, inverse=False):
        x = x.float()  # TODO: revisit when new memcnn version is released
        if inverse:
            return self.invertible_block.inverse(x)
        else:
            return self.invertible_block(x)



class InvertibleSequence(nn.Module):
    def __init__(self, block, n_blocks, keep_input):
        super().__init__()

        sequence = [InvertibleBlock(block, keep_input) for _ in range(n_blocks)]
        self.sequence = nn.Sequential(*sequence)
    
    def forward(self, x, inverse=False):
        if inverse:
            sequence = reversed(self.sequence)
        else:
            sequence = self.sequence
        
        for i, block in enumerate(sequence):
            if i == 0:    #https://github.com/silvandeleemput/memcnn/issues/39#issuecomment-599199122
                if inverse:
                    block.invertible_block.keep_input_inverse = True
                else:
                    block.invertible_block.keep_input = True
            x = block(x, inverse=inverse)
        return x        
