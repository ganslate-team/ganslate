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
    # TODO: make this better
    conf = init_config('./midaGAN/conf/experiment1.yaml')
    cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli)
    print(conf.pretty())

    trainer = Trainer(conf)
    trainer.train()
            
if __name__ == '__main__':
    train()
