import torch
import torch.nn as nn
from pytorch_msssim.ssim import SSIM

class GeneratorLosses:
    def __init__(self, lambda_A, lambda_B, lambda_identity, lambda_inverse, proportion_ssim, **_kwargs):

        self.criterion_cycle = CycleLoss(lambda_A, lambda_B, proportion_ssim)

        if lambda_identity > 0:
            self.criterion_idt = IdentityLoss(lambda_identity, lambda_A, lambda_B)
        else:
            self.criterion_idt = None

        if lambda_inverse > 0:
            self.criterion_inv = InverseLoss(lambda_inverse, lambda_A, lambda_B)
        else:
            self.criterion_inv = None

    def use_identity(self):
        """Check if idt_A and idt_B should be computed."""
        if self.criterion_idt:
            return True
        else:
            return False

    def __call__(self, visuals):
        real_A, real_B = visuals['real_A'], visuals['real_B']
        fake_A, fake_B = visuals['fake_A'], visuals['fake_B']
        rec_A,  rec_B  = visuals['rec_A'],  visuals['rec_B']
        idt_A,  idt_B  = visuals['idt_A'],  visuals['idt_B']
        
        losses = {}

        # cycle-consistency loss
        losses['cycle_A'], losses['cycle_B'] = self.criterion_cycle(real_A, real_B, rec_A, rec_B)

        # identity loss
        if self.criterion_idt:
            if idt_A is not None and idt_B is not None:
                losses['idt_A'], losses['idt_B'] = self.criterion_idt(real_A, real_B, idt_A, idt_B)
            else:
                raise ValueError("idt_A and/or idt_B is not computed but the identity loss is defined.")

        # inverse loss
        if self.criterion_inv is not None:
            losses['inv_A'], losses['inv_B'] = self.criterion_inv(real_A, real_B, fake_A, fake_B)

        return losses

# TODO: Idenity and Inverse are basically the same thing,
# just different args. Make it the same or keep separated for readability?
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


class InverseLoss:
    def __init__(self, lambda_inverse, lambda_A, lambda_B):
        self.lambda_inverse = lambda_inverse
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.criterion = torch.nn.L1Loss()
    
    def __call__(self, real_A, real_B, fake_A, fake_B):
        loss_inv_A = self.criterion(fake_B, real_A)  # || G_A(real_A) - real_A ||
        loss_inv_B = self.criterion(fake_A, real_B)  # || G_B(real_B) - real_B ||

        loss_inv_A = loss_inv_A * self.lambda_A * self.lambda_inverse
        loss_inv_B = loss_inv_B * self.lambda_B * self.lambda_inverse
        
        return loss_inv_A, loss_inv_B

class CycleLoss:
    def __init__(self, lambda_A, lambda_B, proportion_ssim=0):
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

        self.criterion = torch.nn.L1Loss()
        if proportion_ssim > 0:
            # TODO: FIX CHANNEL (refactor ssim code)
            self.ssim_criterion = SSIM(data_range=(-1,1), channel=32)
            # weights for addition of SSIM and L1 losses
            self.alpha = proportion_ssim
            self.beta  = 1 - proportion_ssim 
        else:
            self.ssim_criterion = None
            
    def __call__(self, real_A, real_B, rec_A, rec_B):
        # regular L1 cycle-consistency
        loss_cycle_A = self.criterion(rec_A, real_A)  # || G_B(G_A(real_A)) - real_A||
        loss_cycle_B = self.criterion(rec_B, real_B)  # || G_A(G_B(real_B)) - real_B||

        # cycle-consistency using a weighted combination of SSIM and L1 
        if self.ssim_criterion:
            # (1-SSIM) because the more similar the images are, the higher value will SSIM give (max 1, min -1)
            loss_ssim_A = 1 - self.ssim_criterion(rec_A, real_A) 
            loss_ssim_B = 1 - self.ssim_criterion(rec_B, real_B)

            # weighted sum of SSIM and L1 losses for both forward and backward cycle losses
            loss_cycle_A = self.alpha * loss_ssim_A + self.beta * loss_cycle_A  
            loss_cycle_B = self.alpha * loss_ssim_B + self.beta * loss_cycle_B 

        loss_cycle_A = loss_cycle_A * self.lambda_A 
        loss_cycle_B = loss_cycle_B * self.lambda_B

        return loss_cycle_A, loss_cycle_B