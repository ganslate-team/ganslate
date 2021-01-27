import logging
import sys

logger = logging.getLogger(__name__)

try:
    import midaGAN
except ImportError:
    logger.warning("midaGAN not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import midaGAN

from midaGAN.configs.utils import builders
from midaGAN.inferer import Inferer
from midaGAN.utils import communication, environment


def main():
    # inits distributed mode if ran with torch.distributed.launch
    communication.init_distributed()

    conf = builders.build_inference_conf()
    environment.setup_logging_with_config(conf, mode='inference')

    inferer = Inferer(conf)
    inferer.run()


if __name__ == '__main__':
    main()
