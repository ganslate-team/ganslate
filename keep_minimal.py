import torch
import torch.nn as nn
import memcnn


KEEP = False
EVAL = False

# define a new torch Module with a sequence of operations: Relu o BatchNorm2d o Conv2d
class ExampleOperation(nn.Module):
    def __init__(self, channels):
        super(ExampleOperation, self).__init__()
        self.seq = nn.Sequential(
                                    nn.Conv2d(in_channels=channels, out_channels=channels,
                                              kernel_size=(3, 3), padding=1),
                                    nn.BatchNorm2d(num_features=channels),
                                    nn.ReLU(inplace=True)
                                )

    def forward(self, x):
        return self.seq(x)


# generate some random input data (batch_size, num_channels, y_elements, x_elements)
X = torch.rand(2, 10, 8, 8)
print(X.requires_grad)
#X.requires_grad = True
print(X.requires_grad)

# application of the operation(s) the normal way
model_normal = ExampleOperation(channels=10)
model_normal.eval()

Y = model_normal(X)

# turn the ExampleOperation invertible using an additive coupling
invertible_module = memcnn.AdditiveCoupling(
    Fm=ExampleOperation(channels=10 // 2),
    Gm=ExampleOperation(channels=10 // 2)
)

invertible_module_wrapper = memcnn.InvertibleModuleWrapper(fn=invertible_module, 
                                                           keep_input=KEEP, 
                                                           keep_input_inverse=KEEP)

if EVAL:
    invertible_module_wrapper.eval()
else:
    invertible_module_wrapper.train()


Y2 = invertible_module_wrapper.forward(X)
X2 = invertible_module_wrapper.inverse(Y2)