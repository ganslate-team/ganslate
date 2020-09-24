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

from midaGAN.inferer import Inferer
from midaGAN.conf.builders import build_inference_conf
from midaGAN.utils import communication, environment
# -------------------------------------

def main():
    communication.init_distributed()  # inits distributed mode if ran with torch.distributed.launch

    conf = build_inference_conf()
    #environment.setup_training_logging(conf) TODO: for inference

    inferer = Inferer(conf)
    inferer.run()

if __name__ == '__main__':
    main()
