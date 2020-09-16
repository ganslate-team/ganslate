import sys
import torch
from omegaconf import OmegaConf
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# --------- midaGAN imports ----------
try:
    import midaGAN
except ImportError:
    logger.warning("midaGAN not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import midaGAN

from midaGAN.inferencer import Inferencer
# -------------------------------------

if __name__ == '__main__':
    a = Inferencer()
    a.infer()
