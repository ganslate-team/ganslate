from loguru import logger
import sys

try:
    import midaGAN
except ImportError:
    logger.warning("midaGAN not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import midaGAN

from midaGAN.engines.inferer import Inferer
from midaGAN.utils import communication
from midaGAN.utils.builders import build_conf


def main():
    # inits distributed mode if ran with torch.distributed.launch
    communication.init_distributed()

    conf = build_conf()
    inferer = Inferer(conf)
    inferer.run()


if __name__ == '__main__':
    main()
