try:
    import midaGAN
except ImportError:
    print("midaGAN not installed as a package, importing it from the local directory.")
    import sys
    sys.path.append('./')
    import midaGAN

from midaGAN.trainer import Trainer
from midaGAN.conf import init_config
from omegaconf import OmegaConf

def train():
    cli = OmegaConf.from_cli()
    conf = init_config(cli.config)
    cli.pop("config")

    conf = OmegaConf.merge(conf, cli)
    print(conf.pretty())

    trainer = Trainer(conf)
    trainer.train()
            
if __name__ == '__main__':
    train()
