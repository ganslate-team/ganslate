from omegaconf import OmegaConf

from midaGAN.configs.config import Config
from midaGAN.configs.utils import init_config


def build_conf():
    cli = OmegaConf.from_cli()
    conf = init_config(cli.pop("config"), config_class=Config)
    return OmegaConf.merge(conf, cli)
