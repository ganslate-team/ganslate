import sys
import logging
logger = logging.getLogger(__name__)

# --------- midaGAN imports ----------
try:
    import midaGAN
except ImportError:
    logger.warning("midaGAN not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import midaGAN

from midaGAN.trainer import Trainer
from midaGAN.conf.builders import build_training_conf
from midaGAN.utils import communication, environment


def main():
    communication.init_distributed()  # inits distributed mode if ran with torch.distributed.launch

    conf = build_training_conf()
    environment.setup_logging_with_config(conf)
    
    trainer = Trainer(conf)
    trainer.run()


if __name__ == '__main__':
    main()
