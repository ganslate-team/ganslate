# midaGAN.nn.generators 
from midaGAN.utils import str_to_class

def build_G(conf, device):
    name = conf.generator.name
    generator_class = str_to_class(f"midiGAN.nn.generators.{name.lower()}", name)
    if generator_class is None:
        raise NotImplementedError(f"Generator at `midiGAN.nn.generators.{name.lower()}.{name} not found")
    
    generator_args = dict(conf.generator)
    generator_args.pop("name") # used only to select the model class
    norm_type = conf.gan.norm_type

    generator = generator_class(**generator_args, norm_type=norm_type)
  
    init_type = conf.gan.weight_init_type
    init_gain = conf.gan.weight_init_gain
    init_weights(generator, init_type, init_gain)
    
    return generator.to(device)

