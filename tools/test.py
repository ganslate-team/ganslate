import logging
import sys

logger = logging.getLogger(__name__)

sys.path.append('./')
import midaGAN
#from midaGAN import configs
from midaGAN.utils.builders import build_conf, build_gan
from midaGAN.engines.evaluators import Tester
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
