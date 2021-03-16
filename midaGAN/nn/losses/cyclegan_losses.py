import torch
import midaGAN.nn.losses.utils.ssim as ssim

from loguru import logger


class CycleGANLosses:
    """Defines losses used for optiming the generators in CycleGAN setup.
    Consists of:
        (1) Cycle-consistency loss (weighted combination of L1 and, optionally, SSIM)
        (2) Identity loss
    """

    def __init__(self, conf):
        lambda_A = conf.train.gan.optimizer.lambda_A
        lambda_B = conf.train.gan.optimizer.lambda_B
        lambda_identity = conf.train.gan.optimizer.lambda_identity
        proportion_ssim = conf.train.gan.optimizer.proportion_ssim

        # Cycle-consistency - L1, with optional weighted combination with SSIM
        self.criterion_cycle = CycleLoss(lambda_A, lambda_B, proportion_ssim)
        if lambda_identity > 0:
            self.criterion_idt = IdentityLoss(lambda_identity, lambda_A, lambda_B)
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
        losses['cycle_A'], losses['cycle_B'] = self.criterion_cycle(real_A, real_B, rec_A, rec_B)

        # identity loss
        if self.criterion_idt:
            if idt_A is not None and idt_B is not None:
                losses['idt_A'], losses['idt_B'] = self.criterion_idt(real_A, real_B, idt_A, idt_B)
            else:
                raise ValueError(
                    "idt_A and/or idt_B is not computed but the identity loss is defined.")

        return losses


class CycleLoss:

    def __init__(self, lambda_A, lambda_B, proportion_ssim):
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

        self.criterion = torch.nn.L1Loss()
        if proportion_ssim > 0:
            self.ssim_criterion = ssim.SSIMLoss()

            # weights for addition of SSIM and L1 losses
            self.alpha = proportion_ssim
            self.beta = 1 - proportion_ssim
        else:
            self.ssim_criterion = None

    def __call__(self, real_A, real_B, rec_A, rec_B):
        # regular L1 cycle-consistency
        loss_cycle_A = self.criterion(rec_A, real_A)  # || G_B(G_A(real_A)) - real_A||
        loss_cycle_B = self.criterion(rec_B, real_B)  # || G_A(G_B(real_B)) - real_B||

        # cycle-consistency using a weighted combination of SSIM and L1
        if self.ssim_criterion:
            # Data range needs to be positive and normalized
            # https://github.com/VainF/pytorch-msssim#2-normalized-input
            ssim_real_A = (real_A + 1) / 2
            ssim_real_B = (real_B + 1) / 2

            ssim_rec_A = (rec_A + 1) / 2
            ssim_rec_B = (rec_B + 1) / 2

            # SSIM criterion returns distance metric
            loss_ssim_A = self.ssim_criterion(ssim_rec_A, ssim_real_A, data_range=1)
            loss_ssim_B = self.ssim_criterion(ssim_rec_B, ssim_real_B, data_range=1)

            # weighted sum of SSIM and L1 losses for both forward and backward cycle losses
            loss_cycle_A = self.alpha * loss_ssim_A + self.beta * loss_cycle_A
            loss_cycle_B = self.alpha * loss_ssim_B + self.beta * loss_cycle_B

        loss_cycle_A = loss_cycle_A * self.lambda_A
        loss_cycle_B = loss_cycle_B * self.lambda_B
        return loss_cycle_A, loss_cycle_B


class IdentityLoss:

    def __init__(self, lambda_identity, lambda_A, lambda_B):
        self.lambda_identity = lambda_identity
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.criterion = torch.nn.L1Loss()

    def __call__(self, real_A, real_B, idt_A, idt_B):
        loss_idt_A = self.criterion(idt_A, real_B)  # || G_A(real_B) - real_B ||
        loss_idt_B = self.criterion(idt_B, real_A)  # || G_B(real_A) - real_A ||

        loss_idt_A = loss_idt_A * self.lambda_B * self.lambda_identity
        loss_idt_B = loss_idt_B * self.lambda_A * self.lambda_identity
        return loss_idt_A, loss_idt_B
