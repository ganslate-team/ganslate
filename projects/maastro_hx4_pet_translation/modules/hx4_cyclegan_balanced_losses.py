import math
import torch

from midaGAN.nn.losses import cyclegan_losses


class HX4CycleGANBalancedLosses(cyclegan_losses.CycleGANLosses):
    """ Modified to make Cycle-consitency account for only
    FDG-PET images (in domain A) and HX4-PET images (in domain B),
    and ignore CT components """

    def __init__(self, conf):
        self.lambda_AB = conf.train.gan.optimizer.lambda_AB
        self.lambda_BA = conf.train.gan.optimizer.lambda_BA

        lambda_identity = conf.train.gan.optimizer.lambda_identity
        proportion_ssim = conf.train.gan.optimizer.proportion_ssim

        # Cycle-consistency - L1, with optional weighted combination with SSIM
        self.criterion_cycle = cyclegan_losses.CycleLoss(proportion_ssim)


    def __call__(self, visuals):
        # Separate out the FDG-PET and HX4-PET parts from the visuals tensors 
        real_A1, real_B1 = visuals['real_A'][:, :1], visuals['real_B'][:, :1]
        fake_A1, fake_B1 = visuals['fake_A'][:, :1], visuals['fake_B'][:, :1]
        rec_A1, rec_B1 = visuals['rec_A'][:, :1], visuals['rec_B'][:, :1]

        losses = {}

        # cycle-consistency loss
        losses['cycle_A'] = self.lambda_AB * self.criterion_cycle(real_A1, rec_A1) 
        losses['cycle_B'] = self.lambda_BA * self.criterion_cycle(real_B1, rec_B1)

        return losses