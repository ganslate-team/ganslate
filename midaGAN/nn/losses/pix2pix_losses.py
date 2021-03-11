import torch
import midaGAN.nn.losses.utils.ssim as ssim

import logging
logger = logging.getLogger(__name__)


class L1Loss:
    """Defines Pix2pix specific loss: L1 distance between fake B and real B images
    """
    def __init__(self, conf):
        self.lambda_l1 = conf.train.gan.optimizer.lambda_l1
        self.criterion = torch.nn.L1Loss()

    def __call__(self, fake_B, real_B):
        l1_distance = self.criterion(fake_B, real_B)
        return self.lambda_l1 * l1_distance