"""
If you want to write a custom architecture (GAN, generator, discriminator) or a loss function
it is best to check how they are implemented in ganslate:
https://github.com/Maastro-CDS-Imaging-Group/ganslate/tree/master/ganslate/nn
and to follow the documentation:
https://ganslate.readthedocs.io/en/latest/
"""


# ------------------- Custom GAN from scratch -----------------------
"""Implementing a custom GAN from scratch is not trivial, and we advise you to go
through ganslate's GAN source code for an example.
(https://github.com/Maastro-CDS-Imaging-Group/ganslate/tree/master/ganslate/nn/gans)
"""

# ------------------ Extending an existing GAN ----------------------
"""Extending an existing GAN is much easier. This is an example of how you would start
extending CycleGAN. 
"""
from ganslate.nn.gans.unpaired import cyclegan


@dataclass
class {{cookiecutter.cyclegan_name}}CycleGANConfig(cyclegan.CycleGANConfig):
    pass


class {{cookiecutter.cyclegan_name}}CycleGAN(cyclegan.CycleGAN):
    
    def __init__(self, conf):
        super().__init__(conf)

    # Now, extend or redefine method(s).
    # In this example, we extend only the `init_criterions()` method.
    def init_criterions(self):
        # Standard GAN loss [Same as in the original CycleGAN]
        self.criterion_adv = AdversarialLoss(
            self.conf.train.gan.optimizer.adversarial_loss_type).to(self.device)

        # Fancy loss for generator [Different from the original CycleGAN]
        self.criterion_G = YourFancyLoss(self.conf)


# ------------------ Custom generator or loss -----------------------
"""No limitations, just basic PyTorch code. They do need to have their corresponding configs
as can be seen in the documentation and the source code."""