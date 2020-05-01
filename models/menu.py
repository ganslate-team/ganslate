from models.util import init_weights
from models.generators.vnet import VNet
from models.discriminators.patchgan_discriminator import PatchGANDiscriminator


def define_G(conf, device):
    name = conf.gan.generator.model
    norm_type = conf.gan.norm_type
    
    generator_args = dict(conf.gan.generator)
    generator_args.pop("model") # used only to select the model

    if name.startswith('vnet'):
        generator = VNet(**generator_args, norm_type=norm_type)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % name)
    
    init_type = conf.gan.weight_init_type
    init_gain = conf.gan.weight_init_gain
    init_weights(generator, init_type, init_gain)
    
    return generator.to(device)


def define_D(conf, device):
    name = conf.gan.discriminator.model
    norm_type = conf.gan.norm_type

    discriminator_args = dict(conf.gan.discriminator)
    discriminator_args.pop("model") # used only to select the model

    if name == 'patch_gan':
        discriminator = PatchGANDiscriminator(**discriminator_args, norm_type=norm_type)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % name_D)

    init_type = conf.gan.weight_init_type
    init_gain = conf.gan.weight_init_gain
    init_weights(discriminator, init_type, init_gain)

    return discriminator.to(device)