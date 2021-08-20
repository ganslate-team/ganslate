import torch
import ganslate.nn.losses.utils.ssim as ssim

from loguru import logger


class CycleGANLosses:
    """Defines losses used for optiming the generators in CycleGAN setup.
    Consists of:
        (1) Cycle-consistency loss (weighted combination of L1 and, optionally, SSIM)
        (2) Identity loss
    """

    def __init__(self, conf):
        self.lambda_AB = conf.train.gan.optimizer.lambda_AB
        self.lambda_BA = conf.train.gan.optimizer.lambda_BA

        lambda_identity = conf.train.gan.optimizer.lambda_identity
        proportion_ssim = conf.train.gan.optimizer.proportion_ssim

        # Cycle-consistency - L1, with optional weighted combination with SSIM
        self.criterion_cycle = CycleLoss(proportion_ssim)
        if lambda_identity > 0:
            self.criterion_idt = IdentityLoss(lambda_identity)
        else:
            self.criterion_idt = None

    def is_using_identity(self):
        """Check if idt_A and idt_B should be computed."""
        return True if self.criterion_idt else False

    def __call__(self, visuals):
        real_A, real_B = visuals['real_A'], visuals['real_B']
        fake_A, fake_B = visuals['fake_A'], visuals['fake_B']
        rec_A, rec_B = visuals['rec_A'], visuals['rec_B']
        idt_A, idt_B = visuals['idt_A'], visuals['idt_B']

        losses = {}

        # cycle-consistency loss
        # || G_BA(G_AB(real_A)) - real_A||
        losses['cycle_A'] = self.lambda_AB * self.criterion_cycle(real_A, rec_A) 
        # || G_AB(G_BA(real_B)) - real_B||
        losses['cycle_B'] = self.lambda_BA * self.criterion_cycle(real_B, rec_B)

        # identity loss
        if self.criterion_idt:
            if idt_A is not None and idt_B is not None:
                # || G_AB(real_B) - real_B ||
                losses['idt_B'] = self.lambda_AB * self.criterion_idt(idt_B, real_B)
                # || G_BA(real_A) - real_A ||
                losses['idt_A'] = self.lambda_BA * self.criterion_idt(idt_A, real_A)

            else:
                raise ValueError(
                    "idt_A and/or idt_B is not computed but the identity loss is defined.")

        return losses


class CycleLoss:

    def __init__(self, proportion_ssim):
        self.criterion = torch.nn.L1Loss()
        if proportion_ssim > 0:
            self.ssim_criterion = ssim.SSIMLoss()
            # weights for addition of SSIM and L1 losses
            self.alpha = proportion_ssim
            self.beta = 1 - proportion_ssim
        else:
            self.ssim_criterion = None

    def __call__(self, real, reconstructed):
        # regular L1 cycle-consistency
        cycle_loss_L1 = self.criterion(reconstructed, real)

        # cycle-consistency using a weighted combination of SSIM and L1
        if self.ssim_criterion:
            # Data range needs to be positive and normalized
            # https://github.com/VainF/pytorch-msssim#2-normalized-input
            ssim_real = (real + 1) / 2
            ssim_reconstructed = (reconstructed + 1) / 2

            # SSIM criterion returns distance metric
            cycle_loss_ssim = self.ssim_criterion(ssim_reconstructed, ssim_real, data_range=1)

            # weighted sum of SSIM and L1 losses for both forward and backward cycle losses
            return self.alpha * cycle_loss_ssim + self.beta * cycle_loss_L1
        else:
            return cycle_loss_L1


class IdentityLoss:

    def __init__(self, lambda_identity):
        self.lambda_identity = lambda_identity
        self.criterion = torch.nn.L1Loss()

    def __call__(self, idt, real):
        loss_idt = self.criterion(idt, real)
        return loss_idt * self.lambda_identity
