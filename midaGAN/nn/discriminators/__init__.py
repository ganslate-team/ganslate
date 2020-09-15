
import midaGAN
from midaGAN.utils import import_class_from_dirs_and_modules
from midaGAN.nn.utils import init_weights

def build_D(conf, device):
    # TODO: Abstract away for both G and D?
    name = conf.discriminator.name
    import_locations = midaGAN.conf.IMPORT_LOCATIONS
    discriminator_class = import_class_from_dirs_and_modules(name, import_locations["discriminator"])

    discriminator_args = dict(conf.discriminator)
    discriminator_args.pop("name") # used only to select the model class
    norm_type = conf.gan.norm_type # TODO: interpolate both to G and D from GAN

    discriminator = discriminator_class(**discriminator_args, norm_type=norm_type)
  
    init_type = conf.gan.weight_init_type
    init_gain = conf.gan.weight_init_gain
    init_weights(discriminator, init_type, init_gain)
    
    return discriminator.to(device)
