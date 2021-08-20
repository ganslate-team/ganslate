import math
import torch

from ganslate.nn.losses import cyclegan_losses


class CycleGANLossesForV3(cyclegan_losses.CycleGANLosses):
    """ Modified to make Cycle-consitency account for only
    Normalmap images (in domain A) and depthmap images (in domain B),
    and ignore RGB """

    def __init__(self, conf):
        self.lambda_AB = conf.train.gan.optimizer.lambda_AB
        self.lambda_BA = conf.train.gan.optimizer.lambda_BA

        lambda_identity = conf.train.gan.optimizer.lambda_identity
        proportion_ssim = conf.train.gan.optimizer.proportion_ssim

        # Cycle-consistency - L1, with optional weighted combination with SSIM
        self.criterion_cycle = cyclegan_losses.CycleLoss(proportion_ssim)


    def __call__(self, visuals):
        # Separate out the normalmap and depthmap parts from the visuals tensors 
        real_A2, real_B2 = visuals['real_A'][:, 3:], visuals['real_B'][:, 3:]
        fake_A2, fake_B2 = visuals['fake_A'][:, 3:], visuals['fake_B'][:, 3:]
        rec_A2, rec_B2 = visuals['rec_A'][:, 3:], visuals['rec_B'][:, 3:]

        losses = {}

        # cycle-consistency loss
        losses['cycle_A'] = self.lambda_AB * self.criterion_cycle(real_A2, rec_A2) 
        losses['cycle_B'] = self.lambda_BA * self.criterion_cycle(real_B2, rec_B2)

        return losses
