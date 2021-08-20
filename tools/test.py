from loguru import logger
import sys

try:
    import ganslate
except ImportError:
    logger.warning("ganslate not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import ganslate

#from ganslate import configs
from ganslate.utils.builders import build_conf, build_gan
from ganslate.engines.validator_tester import Tester
from ganslate.utils import communication, environment


def main():
    # inits distributed mode if ran with torch.distributed.launch
    communication.init_distributed()
    environment.setup_threading()

    conf = build_conf()

    tester = Tester(conf)
    tester.run()


if __name__ == '__main__':
    main()
