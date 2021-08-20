from loguru import logger
import sys

try:
    import ganslate
except ImportError:
    logger.warning("ganslate not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import ganslate

from ganslate.engines.trainer import Trainer
from ganslate.utils import communication, environment
from ganslate.utils.builders import build_conf


def main():
    # inits distributed mode if ran with torch.distributed.launch
    communication.init_distributed()
    environment.setup_threading()

    conf = build_conf()

    trainer = Trainer(conf)
    trainer.run()


if __name__ == '__main__':
    main()
