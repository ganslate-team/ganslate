from omegaconf import OmegaConf
from midaGAN.configs.utils import initializers
from midaGAN.configs import inference
import logging

logger = logging.getLogger(__name__)

def build_training_conf():
    cli = OmegaConf.from_cli()
    conf = initializers.init_config(cli.pop("config"))
    return OmegaConf.merge(conf, cli)
