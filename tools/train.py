import logging
import sys

logger = logging.getLogger(__name__)

try:
    import midaGAN
except ImportError:
    logger.warning("midaGAN not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import midaGAN

from midaGAN.engines.trainer import Trainer
from midaGAN.utils import communication, environment
from midaGAN.utils.builders import build_conf


def main():
    # inits distributed mode if ran with torch.distributed.launch
    communication.init_distributed()
    environment.setup_threading()

    conf = build_conf()

    trainer = Trainer(conf)
    trainer.run()


if __name__ == '__main__':
    main()
