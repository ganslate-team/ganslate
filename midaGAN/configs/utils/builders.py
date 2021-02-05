from omegaconf import OmegaConf
from midaGAN.configs.utils import initializers
from midaGAN.configs.config import Config
import logging

logger = logging.getLogger(__name__)

def build_conf():
    cli = OmegaConf.from_cli()
    conf = initializers.init_config(cli.pop("config"), config_class=Config)
    return OmegaConf.merge(conf, cli)
