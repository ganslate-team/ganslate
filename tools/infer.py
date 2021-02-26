import logging
import sys

logger = logging.getLogger(__name__)

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
