import midaGAN
from midaGAN.utils import import_class_from_dirs_and_modules
from midaGAN.nn.utils import init_weights

def build_G(conf, device):

    name = conf.generator.name
    config_locations = midaGAN.conf.CONFIG_LOCATIONS
    generator_class = import_class_from_dirs_and_modules(name, config_locations["generator"])

    generator_args = dict(conf.generator)
    generator_args.pop("name") # used only to select the model class
    norm_type = conf.gan.norm_type

    generator = generator_class(**generator_args, norm_type=norm_type)
  
    init_type = conf.gan.weight_init_type
    init_gain = conf.gan.weight_init_gain
    init_weights(generator, init_type, init_gain)
    
    return generator.to(device)

