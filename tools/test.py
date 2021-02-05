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
from midaGAN.configs._builder import build_conf
from midaGAN.engines.evaluators import Tester
from midaGAN.nn.gans._builder import build_gan
from midaGAN.utils import communication, environment


def main():
    environment.setup_threading()
    # inits distributed mode if ran with torch.distributed.launch
    communication.init_distributed()

    conf = build_conf()

    tester = Tester(conf)
    tester.run()


if __name__ == '__main__':
    main()
