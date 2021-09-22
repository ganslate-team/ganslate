# Writing Your Own GAN Class from Scratch

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
    name: str = "FancyNewGAN"
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
project_dir: projects/your_project
...

train:
    ...

    gan:        
        name: "YourFancyGAN"  # Name of your GAN class   
        ...

    optimizer:                # Optimizer config that includes your custom hyperparameter
            lambda_1: 0.1
            lambda_2: 0.5
            ...
...

```