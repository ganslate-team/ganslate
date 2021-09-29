# Custom Generator or Discriminator Architectures

In image translation GANs, the "generator" can be any network with an architecture that enables accepting as input an image and producing an output image of the same size as the input. Whereas, the discriminator is any network that can take as input these images and produce a real/fake validity score which may either be a scalar or a 2D/3D map with each unit casting a fixed receptive field on the input. In `ganslate`, the generator and discriminator networks are defined as standard _PyTorch_ modules, [constructed by inheriting](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) from the type `torch.nn.Module`. In addition to defining your custom generator or discriminator network, you must also define a configuration dataclass for your network in the same file as follows
```python
from torch import nn
from dataclasses import dataclass
from ganslate import configs

@dataclass
class CustomGeneratorConfig(configs.base.BaseGeneratorConfig):
    name: str = 'CustomGenerator'
    n_residual_blocks: int = 9
    use_dropout: bool = False

class CustomGenerator(nn.Module):
    """Create a custom generator module"""
    def __init__(self, in_channels, out_channels, norm_type, n_residual_blocks, use_dropout):
        # Define the class attributes
        ...
    
    def forward(self, input_tensor):
        # Define the forward pass operation
        ...
```

Ensure that your YAML configuration file includes the pointer to your `CustomGenerator` as well as the appropriate settings
```yaml
project_dir: projects/your_project
...

train:
    ...

    gan:
        ...

        generator:              
            name: "CustomGenerator"  # Name of your custom generator class            
            n_residual_blocks: 9     # Configuration
            in_out_channels:
                AB: [3, 3]
        ...
...
```
