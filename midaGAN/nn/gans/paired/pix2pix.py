import itertools
from dataclasses import dataclass

import torch
from midaGAN import configs
from midaGAN.data.utils.image_pool import ImagePool
from midaGAN.nn.gans.base import BaseGAN
from midaGAN.nn.losses.adversarial_loss import AdversarialLoss


@dataclass
class OptimizerConfig(configs.base.BaseOptimizerConfig):
    """Pix2pix Optimizer Config"""
    # TODO
    pass

@dataclass
class Pix2PixConditionalGANConfig(configs.base.BaseGANConfig):
    """Pix2pix Config"""
    name: str = "Pix2PixConditionalGAN"
    optimizer: OptimizerConfig = OptimizerConfig()


class Pix2PixConditionalGAN(BaseGAN):
    # TODO
    def __init__(self, conf):
        pass
    
    def init_criterions(self):
        pass

    def init_optimizers(self):
        pass

    def set_input(self, input):
        pass

    def optimize_parameters(self):
        pass

    def forward(self):
        pass

    def infer(self, input, cycle='A'):
        pass
