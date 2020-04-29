import torch
import re

from models.util import init_weights
from models.generators.vnet import VNet, DeeperVNet
from models.discriminators.patchGAN_discriminator import PatchGANDiscriminator


def define_G(conf, device=torch.device('cuda:0')):
    keep_input = not conf.gan.generator.use_memory_saving
    name = conf.gan.generator.model

    if name.startswith('pi_vnet'):
        generator = VNet(num_classes=1, keep_input=keep_input)
    elif name.startswith('deeper_vnet'):
        generator = DeeperVNet(num_classes=1, keep_input=keep_input)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % name)
    
    init_type = conf.gan.weight_init_type
    init_gain = conf.gan.weight_init_gain
    init_weights(generator, init_type, init_gain)
    
    return generator.to(device)


def define_D(conf, device=torch.device('cuda:0')):
    name = conf.gan.discriminator.model

    if name == 'patch_gan':
        #TODO: these args bro
        discriminator = PatchGANDiscriminator(**conf.gan.discriminator, norm_type=conf.gan.norm_type, n_channels_input=conf.gan.n_channels_input)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % name_D)

    init_type = conf.gan.weight_init_type
    init_gain = conf.gan.weight_init_gain
    init_weights(discriminator, init_type, init_gain)

    return discriminator.to(device)