from importlib import import_module
import logging

from midaGAN.configs.utils import IMPORT_LOCATIONS
from midaGAN.nn.gans.basegan import BaseGAN
from midaGAN.nn.utils import init_weights
from midaGAN.utils.io import import_class_from_dirs_and_modules

logger = logging.getLogger(__name__)


def build_gan(conf):
    name = conf.train.gan.name
    model_class = import_class_from_dirs_and_modules(name, IMPORT_LOCATIONS["gan"])
    model = model_class(conf)
    return model
