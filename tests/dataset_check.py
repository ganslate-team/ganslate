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
from midaGAN.data import build_loader


def main():
    communication.init_distributed()  # inits distributed mode if ran with torch.distributed.launch

    conf = build_training_conf()
    environment.setup_logging_with_config(conf)
    
    # Load limited entries in the dataloader
    conf.gan.is_train = False 
    data_loader = build_loader(conf)

    for idx in range(10):
        for i, data in enumerate(data_loader):
            print(f"Loading {i}/{len(data_loader)} @ {idx} Pass")

if __name__ == '__main__':
    main()
