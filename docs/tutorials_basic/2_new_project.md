# Your First Project



-------------------------------------
## Creating a Project from a Template

TODO: Cookiecutter stuff



-------------------------------------------------------------------
## Loading Your Own Data into ganslate with Custom Pytorch Datasets

`ganslate` can be run on your own data through creating [your own Pytorch Dataset](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html). 


### Integrating your Pytorch Dataset for Training
Once you have your custom Dataset, it needs to be modified as `ganslate` expects certain structure to make the Dataset class compatible with the framework. 

Namely, it expects *atleast* the following,

1. Specific return format from the `__getitem__` function
2. Dataclass configuration for your dataset

#### Specific return format from the `__getitem__` function
A sample of the format expected to be returned:
```python
def __getitem__(self, index):
    ...

    return {'A': ..., 'B': ...}
```
A dictionary with the following keys is expected to be returned

1. `A`  - corresponds to `torch.Tensor` image (2D/3D) from domain A
2. `B`  - corresponds to `torch.Tensor` image (2D/3D) from domain B


#### Dataclass configuration for your dataset
The Dataset can be dynamically configured through Dataclasses as a part of the [OmegaConf configuration system](https://github.com/omry/omegaconf). Apart from configuration, the Dataclass is also important to allow the framework to easily import your Dataset while training. A sample of this can be found in the [default ImageDataset](https://github.com/Maastro-CDS-Imaging-Group/midaGAN/blob/26564fa721f71c024aa88fb278ecba7de748e55c/midaGAN/data/image_dataset.py#L15) provided with `ganslate`. 

The structure of the Dataclass configuration 
```python
from dataclasses import dataclass
from ganslate import configs

@dataclass
class YourDatasetNameConfig(configs.base.BaseDatasetConfig): # Your dataset always needs to inherit the BaseDatasetConfig
    # Define additional parameters below, these parameters are passed to
    # the dataset and can be used for dynamic configuration. 
    # Examples of parameters
    flip: bool = True

```

`YourDatasetName` is to be consistent with the name of your Pytorch Dataset. The name is also used to import the Dataset module. 


To allow your Dataset to access parameters defined in the Dataclass configuration, the `___init___` function of the Dataset can be modified.
```python
from torch.utils.data import Dataset

class YourDatasetName(Dataset):
    def __init__(self, conf): # `conf` contains the entire configuration yaml 
        self.flip = conf[conf.mode].dataset.flip # Accessing the flip parameter defined in the dataclass

```

#### Importing your Dataset with `ganslate`
Your Dataset along with its Dataclass configuration can be placed in the `projects` folder under a specific project name.
For example,
```
projects/
    your_project/
        your_dataset.py # Contains both the Dataset and Dataclass configuration
        default_docker.yaml
```

Modify the `default_docker.yaml`
```yaml
project: "./projects/your_project" # This needs to point to the directory where your_dataset.py is located
train:
    dataset:
        _target_: project.datasets.YourDatasetName
        root: "<path_to_datadir>" # Path to where the data is 
        # Additional parameters
        flip: True
```

Apart from this, make sure the other parameters in the `default_docker.yaml` are set appropriately. [Refer to configuring your training with yaml files](configuration.md). 

You can now run training with your custom Dataset! Run this command from the root of the repository,
```python
python tools/train.py config=projects/your_project/default_docker.yaml
```



-----------------------------------
## Adding a Custom Loss to CycleGAN

TODO: Edit this section to remove redundant examples

In addition to using out-of-the-box the [popular architectures](https://github.com/Maastro-CDS-Imaging-Group/ganslate/docs/index.md) of GANs and of the generators and discriminators supplied by `ganslate`, you can easily define your custom architectures to suit your specific requirements.

In `ganslate`, a `gan` represents the *system* of generator(s) and discriminator(s) which, during training, includes a set of loss criteria and optimizers, the specification of the flow of data among the generator and discriminator networks during forward pass, the computation of losses, and the update sequence for the generator and discriminator parameters. Depending on your requirements, you can either override one or more of these specific functionalities of the existing GAN classes or write new GAN classes with entirely different architectures.


### Example 1.1. CycleGAN with Custom Losses - Adding a New Loss Component

This example shows how you can modify the default loss criteria of `CycleGAN` to include your custom loss criterion as an _additional_ loss component. This criterion could, for instance, be a *structure-consistency loss* that would constrain the high-level structure of a fake domain `B` image to be similar to that of its corresponding real domain `A` image.

First, create a new file `projects/your_project/architectures/custom_cyclegan.py` and add the following lines. Note that your `CustomCycleGAN1` class must have an associated dataclass as shown.
```python
from dataclasses import dataclass
from ganslate import configs
from ganslate.nn.gans.unpaired import cyclegan


@dataclass 
class OptimizerConfig(configs.base.BaseOptimizerConfig):
    # Define your optimizer parameters specific to your GAN 
    # such as the scaling factor for your custom loss
    lambda_structure_loss: float = 0.1


@dataclass
class CustomCycleGAN1Config(cyclegan.CycleGANConfig):  # Inherit from the `CycleGANConfig` class
    """ Dataclass containing confiuration for your custom CycleGAN """
    name: str = "CustomCycleGAN1"
    optimizer: OptimizerConfig = OptimizerConfig


class CustomCycleGAN1(cyclegan.CycleGAN):  # Inherit from the `CycleGAN` class
    """ CycleGAN with a structure-consistency loss """
    
    def __init__(self, conf):
        # Initialize by invoking the constructor of the parent class
        super().__init__(conf)

    # Now, extend or redefine method(s).
    # In this example, we need to redefine only the `init_criterions` method.
    def init_criterions(self):
        # Standard adversarial loss [Same as in the original CycleGAN]
        self.criterion_adv = AdversarialLoss(
            self.conf.train.gan.optimizer.adversarial_loss_type).to(self.device)

        # Custom set of losses for the generators [Default CycleGAN losses plus your structure-consistency criterion]
        self.criterion_G = CycleGANLossesWithStructure(self.conf)
```

Now, define the `CycleGANLossesWithStructure` by adding the following lines
```python
from ganslate.nn.losses.cyclegan_losses import CycleGANLosses


class CycleGANLossesWithStructure(CycleGANLosses):  # Inherit from the default CycleGAN losses class

    def __init__(self, conf):
        # Invoke the constructor of the parent class to initialize the default loss criteria 
        # such as cycle-consistency and identity (if enabled) losses
        super.__init__(conf)                                                 
        
        # Initialize your structure criterion. 
        # The hyperparameter `lambda_structure_loss` is the scaling factor for this loss component
        lambda_structure_loss = self.conf.train.optimizer.lambda_structure_loss
        self.your_structure_criterion = YourStructureCriterion(lambda_structure_loss)

    def __call__(self, visuals):
        # Invoke the `__call__` method of the parent class to compute the the default CycleGAN losses    
        losses = super.__call__(visuals)
        
        # Compute your custom loss and store as an addiitonal entry in the `losses` dictionary
        real_A = visuals['real_A']
        fake_B = visuals['fake_B']
        losses['your_structure_loss'] = self.your_structure_criterion(real_A, fake_B) 

        return losses
```

Define the class`YourStructureCriterion` that actually implements the structure-consistency criterion 
```python
class YourStructureCriterion():
    def __init__(self, lambda_structure_loss):
        self.lambda_structure_loss = lambda_structure_loss
        # Your structure criterion could be, for instance, an L1 loss, an SSIM loss, 
        # or a custom distance metric
        ...
    
    def __call__(self, real_image, fake_image):
        # Compute the loss and return the scaled value
        ... 
        return self.lambda_structure_loss * loss_value 
```

Finally, edit your YAML configuration file to include the settings for your custom hyperparameter `lambda_structure_loss`
```yaml
project_dir: projects/your_project
...

train:
    ...

    gan:
        name: "CustomCycleGAN1"  # Name of your GAN class 
        ...

        optimizer:               # Optimizer config that includes your custom hyperparameter
            lambda_structure_loss: 0.1
            ...
...

```
Upon starting the training process, `ganslate` will search `your_project` directory for the `CustomCycleGAN1` class and instantiate from it the GAN object with the supplied settings.



### Example 1.2. CycleGAN with Custom Losses - Writing a New Set of CycleGAN Losses

In this example, we seek to not use the default CycleGAN losses at all but instead completely redefine them. The original cycle-consistency criterion involves computing an `L1` loss between the real domain `A` or domain `B`images and their corresponding reconstructed versions. For the sake of this example, let us consider implementing cycle-consistency using a custom distance metric.

Let your custom CycleGAN class be named `CustomCycleGAN2`. Its definition would be mostly the same as that of `CustomCycleGAN1` from _Example 1_. Moving on to the definition of your `CustomCycleGANLosses`, it would be of the following form
```python
class CustomCycleGANLosses(CycleGANLosses):  # Inherit from the default CycleGAN losses class

    def __init__(self, conf):
        # Hyperparameters (here, scaling factors) for your loss
        self.lambda_AB = conf.train.gan.optimizer.lambda_AB
        self.lambda_BA = conf.train.gan.optimizer.lambda_BA
        
        # Instantiate your custom cycle-consistency 
        self.criterion_custom_cycle = CustomCycleLoss()

    def __call__(self, visuals):
        real_A, real_B = visuals['real_A'], visuals['real_B']
        fake_A, fake_B = visuals['fake_A'], visuals['fake_B']
        rec_A, rec_B = visuals['rec_A'], visuals['rec_B']
        idt_A, idt_B = visuals['idt_A'], visuals['idt_B']

        losses = {}

        # Compute cycle-consistency loss        
        losses['cycle_A'] = self.lambda_AB * self.criterion_cycle(real_A, rec_A)  # L_cyc( real_A, G_BA(G_AB(real_A)) )
        losses['cycle_B'] = self.lambda_BA * self.criterion_cycle(real_B, rec_B)  # L_cyc( real_B, G_AB(G_BA(real_B)) )

        return losses
        
        
class CustomCycleLoss():

    def __init__(self, proportion_ssim):
        ...

    def __call__(self, real, reconstructed):
        # Your alternate formulation of the cycle-consistency criterion
        ...
        return custom_cycle_loss
```
