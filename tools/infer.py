import logging
import sys

logger = logging.getLogger(__name__)

try:
    import midaGAN
except ImportError:
    logger.warning("midaGAN not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import midaGAN

from midaGAN.configs._builder import build_conf
from midaGAN.inferer import Inferer
from midaGAN.utils import communication, environment


def parse_input(input_path):
    """This function should:
        (1) Read the input
        (2) Preprocess it as done in the training phase
    Since the step 2 can vary greatly, it needs to be specific to each project,
    so the user has to implement it on their own if they wish to deploy the model
    for inference.
    """
    raise NotImplementedError()


def main():
    # inits distributed mode if ran with torch.distributed.launch
    communication.init_distributed()

    conf = build_conf()
    environment.setup_logging_with_config(conf)

    inferer = Inferer(conf)
    inferer.run()


if __name__ == '__main__':
    main()
