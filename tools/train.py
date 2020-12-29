import logging
import sys

logger = logging.getLogger(__name__)

try:
    import midaGAN
except ImportError:
    logger.warning("midaGAN not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import midaGAN

#from midaGAN import configs
from midaGAN.configs.utils import builders
from midaGAN.trainer import Trainer
from midaGAN.utils import communication, environment


def main():
    environment.threading_setup()
    communication.init_distributed()  # inits distributed mode if ran with torch.distributed.launch

    conf = builders.build_training_conf()
    environment.setup_logging_with_config(conf)

    trainer = Trainer(conf)
    trainer.run()


if __name__ == '__main__':
    main()
