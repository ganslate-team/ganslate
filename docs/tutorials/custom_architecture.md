# Defining Your Own GAN and Network Architectures

In addition to using out-of-the-box the [popular architectures](https://github.com/Maastro-CDS-Imaging-Group/ganslate/docs/index.md) of GANs and of the generators and discriminators supplied by `ganslate`, you can easily define your custom architectures to suit your specific requirements.


------------------------------
## 1. Custom GAN Architectures

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
    _target_: str = "CustomCycleGAN1"
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
project: projects/your_project
...

train:
    ...

    gan:
        _target_: "project.architectures.CustomCycleGAN1"  # Location of your GAN class 
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



### Example 1.3. Writing Your Own GAN Class from Scratch

Advanced users may opt to implement a new GAN architecture from scratch. You can do this by inheriting from the abstract base class `BaseGAN` and implementing the required methods. All the existing GAN architectures in `ganslate` are defined in this manner. The file containing your `FancyNewGAN` must be structured as follows

```python
from dataclasses import dataclass
from ganslate import configs
from ganslate.nn.gans.base import BaseGAN

@dataclass
class OptimizerConfig(configs.base.BaseOptimizerConfig):
    # Define your optimizer parameters specific to your GAN 
    # such as the scaling factors for a custom loss 
    lambda_1: float = 0.1
    lambda_2: float = 0.5


@dataclass
class FancyNewGANConfig(configs.base.BaseGANConfig):
    # Configuration dataclass for your GAN
    _target_: str = "FancyNewGAN"
    optimizer: OptimizerConfig = OptimizerConfig


class FancyNewGAN(BaseGAN):
    def __init__(self, conf):
        # Constructor for your GAN class
    
    def init_criterions(self):
        # To initialize criterions (losses)
    
    def init_optimizers(self):
        # To initialize optimizers
    
    def set_input(self):
        # To supply input images to the GAN
    
    def forward(self):
        # To perform forward pass through the generator(s) and compute the fake images

    def optimize_parameters(self):
        # To compute losses and perform back-propagation
```

To provide a concrete example of how each of these methods should be defined, let us consider the case of a simple GAN architecture like `Pix2PixConditionalGAN` ([source](https://github.com/Maastro-CDS-Imaging-Group/ganslate/blob/documentation/ganslate/nn/gans/paired/pix2pix.py)) and look at the definition of its methods.

1. The constructor method: Initializes the GAN system with the given configuration.
```python
def __init__(self, conf):
    # The constructor method `__init__` must first invoke the constructor 
    # of the parent class `BaseGAN` to initialize the default attributes. 
    super.__init__(conf)

    # Then, you must define four dictionaries: 
    #       `self.visuals`, `self.losses`, `self.networks`, and `self.optimizers`.
    # `self.visuals` would be used to store image tensors. Set the values to `None` for now.
    self.visual_names = ['real_A', 'real_B', 'fake_B']
    self.visuals = {name: None for name in visual_names}

    # `self.losses` stores the values of various losses used by the model. 
    # Set the values to `None` for now.
    loss_names = ['G', 'D', 'pix2pix']
    self.losses = {name: None for name in loss_names}

    # `self.networks` would contain the generator and discriminator nets used by the GAN. 
    # Set the values to `None` for now.
    network_names = ['G', 'D'] if self.is_train else ['G']
    self.networks = {name: None for name in network_names}

    # With `self.optimizers`, you can define each optimizer for one network or for multiple networks 
    # (for instance, when multiple generators and  discriminators are present like in CycleGAN). 
    # Here, for Pix2Pix, we define one optimizer for the generator and one for the discriminator. 
    # Set the values to `None` for now.
    optimizer_names = ['G', 'D']
    self.optimizers = {name: None for name in optimizer_names}

    # Invoke the `setup` method of the `BaseGAN` parent class to intialize the loss criterions, 
    # networks, optimizer, and to set up mixed precision, checkpoint loading, network parallelization, etc. 
    self.setup()
```

2. Initialization methods: Responsible for initializing the various components of the GAN system. You must implemenent the two initialization methods `init_criterions` and `init_optimizers`. Other such methods exist including `init_networks`, `init_schedulers`, and `init_metrics` which are all internally invoked by the `setup` method to populate their corresponding attributes with initial values. However, these latter three are predefined in the parent `BaseGAN` class and need not be altered in most cases.
```python
def init_criterions(self):
    # Standard GAN (adversarial) loss
    self.criterion_adv = AdversarialLoss(self.conf.train.gan.optimizer.adversarial_loss_type).to(self.device)
    # Pixelwise L1 loss
    self.criterion_pix2pix = Pix2PixLoss(self.conf).to(self.device)

def init_optimizers(self):
    # Access the optimizer parameters in the `config` object
    lr_G = self.conf.train.gan.optimizer.lr_G
    lr_D = self.conf.train.gan.optimizer.lr_D
    beta1 = self.conf.train.gan.optimizer.beta1
    beta2 = self.conf.train.gan.optimizer.beta2

    # Initialize the optimzers as `torch.optim.Adam` objects with the given parameters
    self.optimizers['G'] = torch.optim.Adam(
        self.networks['G'].parameters(), lr=lr_G, betas=(beta1, beta2))
    self.optimizers['D'] = torch.optim.Adam(
        self.networks['D'].parameters(), lr=lr_D, betas=(beta1, beta2))
```

3. The `set_input` method: Accepts the data dictionary supplied by the dataloader, unpacks it, and stores the image tensors for further usage. Called in every training iteration as well as during inference.
```python
def set_input(self, input_dict):
    # The argument `input_dict` is a dictionary obtained from the dataloader, 
    # which contains pair of data samples from domain A and domain B.
    
    # Unpack input data
    self.visuals['real_A'] = input_dict['A'].to(self.device)
    self.visuals['real_B'] = input_dict['B'].to(self.device)
```

4. The `forward` method: Implements forward pass through the generator(s). This method is called by both methods `optimize_parameters` and `test`. Called in every training iteration as well as during inference.
```python
def forward(self):
    # Run forward pass.
    real_A = self.visuals['real_A']      # A
    fake_B = self.networks['G'](real_A)  # G(A)

    # Store the computed fake domain B image
    self.visuals.update({'fake_B': fake_B})
```

5. The `optimize_parameters` method: Implements forward pass through the discriminator(s), loss computation, back-propagation, and the parameter update sequence. Called in every training iteration.
```python
def optimize_parameters(self):
    # Compute fake images
    self.forward()

    # Compute generator based metrics dependent on visuals
    self.metrics.update(self.training_metrics.compute_metrics_G(self.visuals))

    # ------------------------ G ------------------------
    # D requires no gradients when optimizing G
    self.set_requires_grad(
        self.networks['D'], False)
    self.optimizers['G'].zero_grad(set_to_none=True)
    # Calculate gradients for G. Loss computation and back-prop are 
    # abstracted away into a separate method `backward_G`
    self.backward_G()
    # Update G's weights
    self.optimizers['G'].step()

    # ------------------------ D ------------------------
    self.set_requires_grad(self.networks['D'], True)
    self.optimizers['D'].zero_grad(set_to_none=True)
    # Calculate gradients for D. Loss computation and back-prop are 
    # abstracted away into a separate method `backward_D`
    self.backward_D()

    # Update metrics for D
    self.metrics.update(
        self.training_metrics.compute_metrics_D('D', self.pred_real, self.pred_fake))

    self.optimizers['D'].step()     # update D's weights
```

The `backward_G` method here calculates the loss for generator `G` using all specified losses as well as their gradients, and can be defined as
```python
def backward_G(self):
    real_A = self.visuals['real_A']  # A
    real_B = self.visuals['real_B']  # B
    fake_B = self.visuals['fake_B']  # G(A)

    # ------------------------- GAN Loss --------------------------
    # Compute D(A, G(A)) and calculate the adversarial loss
    pred = self.networks['D'](torch.cat([real_A, fake_B], dim=1))    
    self.losses['G'] = self.criterion_adv(pred, target_is_real=True)

    # ------------------------ Pix2Pix Loss -----------------------
    # Calculate the pixel-wise loss
    self.losses['pix2pix'] = self.criterion_pix2pix(fake_B, real_B)

    # Combine losses and calculate gradients with back-propagation
    combined_loss_G = self.losses['G'] + self.losses['pix2pix']
    self.backward(loss=combined_loss_G, optimizer=self.optimizers['G'])
```
Whereas, the `backward_D` method calculates the adversarial loss for the discriminator `D` as well as their gradients, and is defined as
```python
def backward_D(self):
    real_A = self.visuals['real_A']  # A
    real_B = self.visuals['real_B']  # B
    fake_B = self.visuals['fake_B']  # G(A)

    # Discriminator prediction with real data [i.e. D(A, B)]
    self.pred_real = self.networks['D'](
        torch.cat([real_A, real_B], dim=1))

    # Discriminator prediction with fake data [i.e. # D(A, G(A))]
    self.pred_fake = self.networks['D'](
        torch.cat([real_A, fake_B.detach()], dim=1))                  

    # Calculate the adversarial loss for `D`
    loss_real = self.criterion_adv(self.pred_real, target_is_real=True)
    loss_fake = self.criterion_adv(self.pred_fake, target_is_real=False)
    self.losses['D'] = loss_real + loss_fake

    # Compute gradients
    self.backward(loss=self.losses['D'], optimizer=self.optimizers['D'])
```

The aforementioned methods are to be mandatorily implemented if you wish to contruct your own GAN architecture in `ganslate` from scratch. Additionally, We also recommend referring to the abstract `BaseGAN` class ([source](https://github.com/Maastro-CDS-Imaging-Group/ganslate/blob/documentation/ganslate/nn/gans/base.py)) to get an overview of other existing methods and of the internal logic. Finally, update your YAML configuration file to include the apporapriate settings for your custom-defined components
```yaml
project: projects/your_project
...

train:
    ...

    gan:        
        _target_: "project.architectures.YourFancyGAN"  # Location of your GAN class   
        ...

    optimizer:                # Optimizer config that includes your custom hyperparameter
            lambda_1: 0.1
            lambda_2: 0.5
            ...
...

```


-----------------------------------------------------
## 2. Custom Generator or Discriminator Architectures

In image translation GANs, the "generator" can be any network with an architecture that enables taking as input an image and producing an output image of the same size as the input. Whereas, the discriminator is any network that can take as input these images and produce a real/fake validity score which may either be a scalar or a 2D/3D map with each unit casting a fixed receptive field on the input. In `ganslate`, the generator and discriminator networks are defined as standard _PyTorch_ modules, [constructed by inheriting](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) from the type `torch.nn.Module`. In addition to defining your custom generator or discriminator network, you must also define a configuration dataclass for your network in the same file as follows
```python
from torch import nn
from dataclasses import dataclass
from ganslate import configs

@dataclass
class CustomGeneratorConfig(configs.base.BaseGeneratorConfig):
    _target_: str = 'CustomGenerator'
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
project: projects/your_project
...

train:
    ...

    gan:
        ...

        generator:              
            _target_: "project.architectures.CustomGenerator"  # Location of your custom generator class            
            n_residual_blocks: 9     # Configuration
            in_out_channels:
                AB: [3, 3]
        ...
...
```
