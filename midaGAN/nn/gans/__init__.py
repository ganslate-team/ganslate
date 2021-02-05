from importlib import import_module
import logging

import midaGAN
from midaGAN.nn.gans.basegan import BaseGAN
from midaGAN.nn.utils import init_weights
from midaGAN.utils import import_class_from_dirs_and_modules

logger = logging.getLogger(__name__)


def build_gan(conf):
    name = conf.train.gan.name
    import_locations = midaGAN.configs.utils.initializers.IMPORT_LOCATIONS
    model_class = import_class_from_dirs_and_modules(name, import_locations["gan"])
    model = model_class(conf)
    return model
