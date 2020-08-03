from importlib import import_module
from midaGAN.nn.base_model import BaseModel
from midaGAN.nn.utils import init_weights

# TODO: do these import better
from midaGAN.nn.generators.vnet import VNet
from midaGAN.nn.generators.vnet2d import VNet2D
from midaGAN.nn.discriminators.patchgan_discriminator import PatchGANDiscriminator
from midaGAN.nn.discriminators.patchgan2d_discriminator import PatchGAN2DDiscriminator

def build_model(conf):
    model = find_model_using_name(conf.gan.model)
    return model(conf)

def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "nn/modelname_model.py"
    # will be imported.
    model_filename = "midaGAN.nn." + model_name + "_model"
    modellib = import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def build_G(conf, device):
    name = conf.gan.generator.model
    norm_type = conf.gan.norm_type
    
    generator_args = dict(conf.gan.generator)
    generator_args.pop("model") # used only to select the model

    if name.startswith('vnet2d'):
        generator = VNet2D(**generator_args, norm_type=norm_type)
    elif name.startswith('vnet'):
        generator = VNet(**generator_args, norm_type=norm_type)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % name)
    
    init_type = conf.gan.weight_init_type
    init_gain = conf.gan.weight_init_gain
    init_weights(generator, init_type, init_gain)
    
    return generator.to(device)


def build_D(conf, device):
    name = conf.gan.discriminator.model
    norm_type = conf.gan.norm_type

    discriminator_args = dict(conf.gan.discriminator)
    discriminator_args.pop("model") # used only to select the model

    if name == 'patchgan2d':
        discriminator = PatchGAN2DDiscriminator(**discriminator_args, norm_type=norm_type)
    elif name == 'patchgan':
        discriminator = PatchGANDiscriminator(**discriminator_args, norm_type=norm_type)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % name_D)

    init_type = conf.gan.weight_init_type
    init_gain = conf.gan.weight_init_gain
    init_weights(discriminator, init_type, init_gain)

    return discriminator.to(device)
