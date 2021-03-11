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
from midaGAN.utils.builders import build_conf, build_gan
from midaGAN.engines.validator_tester import Tester
from midaGAN.utils import communication, environment


def main():
    # inits distributed mode if ran with torch.distributed.launch
    communication.init_distributed()
    environment.setup_threading()

    conf = build_conf()

    tester = Tester(conf)
    tester.run()


if __name__ == '__main__':
    main()
