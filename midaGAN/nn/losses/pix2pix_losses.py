import torch
import midaGAN.nn.losses.utils.ssim as ssim

import logging
logger = logging.getLogger(__name__)


class Pix2PixLoss:
    """Defines "pixel-to-pixel" loss (applied voxel-to-voxel for 3D omages) 
    L1 distance between fake_B and real_B images
    """
    def __init__(self, conf):
        self.lambda_l1 = conf.train.gan.optimizer.lambda_l1
        self.criterion = torch.nn.L1Loss()

    def __call__(self, fake_B, real_B):
        pix2pix_loss = self.criterion(fake_B, real_B)
        return self.lambda_l1 * pix2pix_loss