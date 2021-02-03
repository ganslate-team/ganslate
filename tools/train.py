import logging
import sys

logger = logging.getLogger(__name__)

try:
    import midaGAN
except ImportError:
    logger.warning("midaGAN not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import midaGAN
    
from midaGAN import configs
from midaGAN.engines.trainer import Trainer
from midaGAN.utils import communication, environment


def main():
    environment.setup_threading()
    # inits distributed mode if ran with torch.distributed.launch
    communication.init_distributed()

    conf = configs.utils.builders.build_conf()

    trainer = Trainer(conf)
    trainer.run()


if __name__ == '__main__':
    main()
