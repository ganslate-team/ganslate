import torch
import re

from models.util import init_weights
from models.generators.vnet import VNet, DeeperVNet
from models.discriminators.patchGAN_discriminator import NLayerDiscriminator
from models.discriminators.pixel_discriminator import PixelDiscriminator


def define_G(conf, device=torch.device('cuda:0')):
    keep_input = not conf.model.use_memory_saving
    name = conf.model.model_G

    if name.startswith('vnet'):
        generator = VNet(num_classes=1, keep_input=keep_input)
    elif name.startswith('deeper_vnet'):
        generator = DeeperVNet(num_classes=1, keep_input=keep_input)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % name_G)
    
    init_type = conf.model.weight_init_type
    init_gain = conf.model.weight_init_gain
    init_weights(generator, init_type, init_gain)
    
    return generator.to(device)


def define_D(conf, device=torch.device('cuda:0')):
    use_sigmoid = conf.model.no_lsgan
    name = conf.model.model_D

    if name == 'n_layers':
        discriminator = NLayerDiscriminator(**conf.model, use_sigmoid=use_sigmoid) # TODO: sigmoid what
    elif name == 'pixel':
        discriminator = PixelDiscriminator(**conf.model, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % name_D)

    init_type = conf.model.weight_init_type
    init_gain = conf.model.weight_init_gain
    init_weights(discriminator, init_type, init_gain)

    return discriminator.to(device)