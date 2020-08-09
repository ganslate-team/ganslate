from importlib import import_module
import logging

from midaGAN.nn.gans.basegan import BaseGAN
from midaGAN.nn.utils import init_weights

logger = logging.getLogger(__name__)

def build_gan(conf):
    model = find_model_using_name(conf.gan.model)
    return model(conf)

def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "nn/modelname_model.py"
    # will be imported.
    model_filename = "midaGAN.nn.gans." + model_name
    modellib = import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseGAN,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseGAN):
            model = cls

    if model is None:
        logger.error("In %s.py, there should be a subclass of BaseGAN with class name that \
                      matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model
