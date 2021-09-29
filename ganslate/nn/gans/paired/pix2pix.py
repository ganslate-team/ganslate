import itertools
from dataclasses import dataclass

import torch
from ganslate import configs
from ganslate.nn.gans.base import BaseGAN
from ganslate.nn.losses.adversarial_loss import AdversarialLoss
from ganslate.nn.losses.pix2pix_losses import Pix2PixLoss


@dataclass
class OptimizerConfig(configs.base.BaseOptimizerConfig):
    """Pix2pix Optimizer Config"""
    lambda_pix2pix: float = 100.0


@dataclass
class Pix2PixConditionalGANConfig(configs.base.BaseGANConfig):
    """Pix2pix Config"""
    optimizer: OptimizerConfig = OptimizerConfig()


class Pix2PixConditionalGAN(BaseGAN):

    def __init__(self, conf):
        super().__init__(conf)

        # Inputs and Outputs of the model
        visual_names = ['real_A', 'fake_B', 'real_B']
        # initialize the visuals as None
        self.visuals = {name: None for name in visual_names}

        # Losses used by the model
        loss_names = ['G', 'D', 'pix2pix']
        self.losses = {name: None for name in loss_names}

        # Optimizers
        optimizer_names = ['G', 'D']
        self.optimizers = {name: None for name in optimizer_names}

        # Generators and Discriminators
        network_names = ['G', 'D'] if self.is_train else ['G']
        self.networks = {name: None for name in network_names}

        # Set up networks, optimizers, schedulers, mixed precision, checkpoint loading, network parallelization...
        self.setup()

    def init_criterions(self):
        # Standard GAN loss
        self.criterion_adv = AdversarialLoss(
            self.conf.train.gan.optimizer.adversarial_loss_type).to(self.device)
        # Pixelwise loss
        self.criterion_pix2pix = Pix2PixLoss(self.conf)

    def init_optimizers(self):
        lr_G = self.conf.train.gan.optimizer.lr_G
        lr_D = self.conf.train.gan.optimizer.lr_D
        beta1 = self.conf.train.gan.optimizer.beta1
        beta2 = self.conf.train.gan.optimizer.beta2

        self.optimizers['G'] = torch.optim.Adam(self.networks['G'].parameters(),
                                                lr=lr_G,
                                                betas=(beta1, beta2))
        self.optimizers['D'] = torch.optim.Adam(self.networks['D'].parameters(),
                                                lr=lr_D,
                                                betas=(beta1, beta2))

    def set_input(self, input):
        """Unpack input data from the dataloader.
        Parameters:
            input (dict) -- a pair of data samples from domain A and domain B.
        """
        self.visuals['real_A'] = input['A'].to(self.device)
        self.visuals['real_B'] = input['B'].to(self.device)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights. 
        Called in every training iteration.
        """
        self.forward()  # compute fake images

        # Compute generator based metrics dependent on visuals
        self.metrics.update(self.training_metrics.compute_metrics_G(self.visuals))

        # ------------------------ G ------------------------
        self.set_requires_grad(self.networks['D'],
                               False)  # D requires no gradients when optimizing G
        self.optimizers['G'].zero_grad(set_to_none=True)
        self.backward_G()  # calculate gradients for G
        self.optimizers['G'].step()  # update G's weights

        # ------------------------ D ------------------------
        self.set_requires_grad(self.networks['D'], True)
        self.optimizers['D'].zero_grad(set_to_none=True)
        self.backward_D()  # calculate gradients for D

        # Update metrics for D
        self.metrics.update(
            self.training_metrics.compute_metrics_D('D', self.pred_real, self.pred_fake))

        self.optimizers['D'].step()  # update D's weights

    def backward_G(self):
        """Calculate the loss for generator G using all specified losses"""

        real_A = self.visuals['real_A']  # A
        real_B = self.visuals['real_B']  # B
        fake_B = self.visuals['fake_B']  # G(A)

        # ------------------------- GAN Loss ----------------------------
        pred = self.networks['D'](torch.cat([real_A, fake_B], dim=1))  # D(A, G(A))
        self.losses['G'] = self.criterion_adv(pred, target_is_real=True)
        # ---------------------------------------------------------------

        # --------------------- Pix2Pix Loss  --------------------
        self.losses['pix2pix'] = self.criterion_pix2pix(fake_B, real_B)
        # --------------------------------------------------------

        # combine losses and calculate gradients
        combined_loss_G = self.losses['G'] + self.losses['pix2pix']
        self.backward(loss=combined_loss_G, optimizer=self.optimizers['G'])

    def backward_D(self):
        """Calculate GAN loss for the discriminator
        Compute the discriminator loss.
        Also calls backward() on loss_D to calculate the gradients.
        """
        real_A = self.visuals['real_A']  # A
        real_B = self.visuals['real_B']  # B
        fake_B = self.visuals['fake_B']  # G(A)

        self.pred_real = self.networks['D'](torch.cat([real_A, real_B], dim=1))  # D(A, G(A))

        # Detaching fake: https://github.com/pytorch/examples/issues/116
        self.pred_fake = self.networks['D'](torch.cat([real_A, fake_B.detach()],
                                                      dim=1))  # D(A, G(A))

        loss_real = self.criterion_adv(self.pred_real, target_is_real=True)
        loss_fake = self.criterion_adv(self.pred_fake, target_is_real=False)
        self.losses['D'] = loss_real + loss_fake

        # backprop
        self.backward(loss=self.losses['D'], optimizer=self.optimizers['D'])

    def forward(self):
        """Run forward pass; called by both methods <optimize_parameters> and <test>."""
        real_A = self.visuals['real_A']  # A

        # Forward G
        fake_B = self.networks['G'](real_A)  # G(A)

        self.visuals.update({'fake_B': fake_B})

    def infer(self, input):
        with torch.no_grad():
            return self.networks['G'].forward(input)
