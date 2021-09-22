import itertools
from dataclasses import dataclass

import torch
from ganslate import configs
from ganslate.data.utils.image_pool import ImagePool
from ganslate.nn.gans.unpaired import cyclegan
from ganslate.nn.gans.base import BaseGAN
from ganslate.nn.losses.adversarial_loss import AdversarialLoss
from ganslate.nn.losses.cyclegan_losses import CycleGANLosses


@dataclass
class OptimizerConfig(cyclegan.OptimizerConfig):
    # the same as CycleGAN, kept here for consistency and for possible future changes
    pass


@dataclass
class RevGANConfig(configs.base.BaseGANConfig):
    """RevGAN Config"""
    pool_size: int = 50
    optimizer: OptimizerConfig = OptimizerConfig


class RevGAN(BaseGAN):
    """ RevGAN architecture, described in the paper
    `Reversible GANs for Memory-efficient Image-to-Image Translation`,
    Tycho F.A. van der Ouderaa, Daniel E. Worrall,
    CVPR 2019
    """

    def __init__(self, conf):
        super().__init__(conf)

        # Inputs and Outputs of the model
        visual_names = ['real_A', 'fake_B', 'rec_A', 'idt_A', 'real_B', 'fake_A', 'rec_B', 'idt_B']
        # initialize the visuals as None
        self.visuals = {name: None for name in visual_names}

        # Losses used by the model
        loss_names = ['G_AB', 'D_B', 'cycle_A', 'idt_A', 'G_BA', 'D_A', 'cycle_B', 'idt_B']
        self.losses = {name: None for name in loss_names}

        # Optimizers
        optimizer_names = ['G', 'D']
        self.optimizers = {name: None for name in optimizer_names}

        # Generator and Discriminators
        network_names = ['G', 'D_B', 'D_A'] if self.is_train else ['G']
        self.networks = {name: None for name in network_names}

        if self.is_train:
            # Create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(conf.train.gan.pool_size)
            self.fake_B_pool = ImagePool(conf.train.gan.pool_size)

        # Set up networks, optimizers, schedulers, mixed precision, checkpoint loading, network parallelization...
        self.setup()

    def init_criterions(self):
        # Standard GAN loss
        self.criterion_adv = AdversarialLoss(
            self.conf.train.gan.optimizer.adversarial_loss_type).to(self.device)
        # Generator-related losses -- Cycle-consistency and Identity loss
        self.criterion_G = CycleGANLosses(self.conf)

    def init_optimizers(self):
        lr_G = self.conf.train.gan.optimizer.lr_G
        lr_D = self.conf.train.gan.optimizer.lr_D
        beta1 = self.conf.train.gan.optimizer.beta1
        beta2 = self.conf.train.gan.optimizer.beta2

        params_G = self.networks['G'].parameters()
        params_D = itertools.chain(self.networks['D_B'].parameters(),
                                   self.networks['D_A'].parameters())

        self.optimizers['G'] = torch.optim.Adam(params_G, lr=lr_G, betas=(beta1, beta2))
        self.optimizers['D'] = torch.optim.Adam(params_D, lr=lr_D, betas=(beta1, beta2))

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
        discriminators = [self.networks['D_B'], self.networks['D_A']]

        self.forward()  # compute fake images and reconstruction images.

        # Compute generator based metrics dependent on visuals
        self.metrics.update(self.training_metrics.compute_metrics_G(self.visuals))

        # ------------------------ G (A and B) ----------------------------------------------------
        self.set_requires_grad(discriminators, False)  # Ds require no gradients when optimizing Gs
        self.optimizers['G'].zero_grad(set_to_none=True)
        self.backward_G()  # calculate gradients for G
        self.optimizers['G'].step()  # update G's weights
        # ------------------------ D_B and D_A ----------------------------------------------------
        self.set_requires_grad(discriminators, True)
        self.optimizers['D'].zero_grad(set_to_none=True)
        self.backward_D('D_B')  # calculate gradients for D_B

        # Update metrics for D_B
        self.metrics.update(
            self.training_metrics.compute_metrics_D('D_B', self.pred_real, self.pred_fake))

        self.backward_D('D_A')  # calculate gradients for D_A

        # Update metrics for D_A
        self.metrics.update(
            self.training_metrics.compute_metrics_D('D_A', self.pred_real, self.pred_fake))

        self.optimizers['D'].step()  # update D_B and D_A's weights
        # -----------------------------------------------------------------------------------------

    def forward(self):
        """Run forward pass; called by both methods <optimize_parameters> and <test>."""
        real_A = self.visuals['real_A']
        real_B = self.visuals['real_B']

        # Forward cycle G_AB (A to B)
        fake_B = self.networks['G'](real_A)  # G_AB
        rec_A = self.networks['G'](fake_B, inverse=True)  # G_BA

        # Backward cycle G_BA (B to A)
        fake_A = self.networks['G'](real_B, inverse=True)  # G_BA
        rec_B = self.networks['G'](fake_A)  # G_AB

        # Visuals for Identity loss
        idt_B, idt_A = None, None
        if self.criterion_G.is_using_identity():
            idt_B = self.networks['G'](real_B)
            idt_A = self.networks['G'](real_A, inverse=True)

        self.visuals.update({
            'fake_B': fake_B,
            'rec_A': rec_A,
            'idt_A': idt_A,
            'fake_A': fake_A,
            'rec_B': rec_B,
            'idt_B': idt_B
        })

    def backward_D(self, discriminator):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        Also calls backward() on loss_D to calculate the gradients.
        """
        if discriminator == 'D_B':
            real = self.visuals['real_B']
            fake = self.visuals['fake_B']
            fake = self.fake_B_pool.query(fake)
            loss_id = 0

        elif discriminator == 'D_A':
            real = self.visuals['real_A']
            fake = self.visuals['fake_A']
            fake = self.fake_A_pool.query(fake)
            loss_id = 1
        else:
            raise ValueError('The discriminator has to be either "D_A" or "D_B".')

        self.pred_real = self.networks[discriminator](real)
        self.pred_fake = self.networks[discriminator](fake.detach())

        loss_real = self.criterion_adv(self.pred_real, target_is_real=True)
        loss_fake = self.criterion_adv(self.pred_fake, target_is_real=False)
        self.losses[discriminator] = loss_real + loss_fake

        # backprop
        self.backward(loss=self.losses[discriminator],
                      optimizer=self.optimizers['D'],
                      retain_graph=True,
                      loss_id=loss_id)

    def backward_G(self):
        """Calculate the loss for generators G_AB and G_BA using all specified losses"""

        fake_B = self.visuals['fake_B']  # G_AB(A)
        fake_A = self.visuals['fake_A']  # G_BA(B)

        # ------------------------- GAN Loss ----------------------------
        pred_B = self.networks['D_B'](fake_A)  # D_B(G_AB(A))
        pred_A = self.networks['D_A'](fake_B)  # D_A(G_BA(B))

        # Forward GAN loss D_A(G_AB(A))
        self.losses['G_AB'] = self.criterion_adv(pred_B, target_is_real=True)
        # Backward GAN loss D_B(G_BA(B))
        self.losses['G_BA'] = self.criterion_adv(pred_A, target_is_real=True)
        # ---------------------------------------------------------------

        # ------------- G Losses (Cycle, Identity) -------------
        losses_G = self.criterion_G(self.visuals)
        self.losses.update(losses_G)
        # ---------------------------------------------------------------

        # combine losses and calculate gradients
        combined_loss_G = sum(losses_G.values()) + self.losses['G_AB'] + self.losses['G_BA']
        self.backward(loss=combined_loss_G, optimizer=self.optimizers['G'], loss_id=2)

    def infer(self, input, direction='AB'):
        assert direction in ['AB', 'BA'], "Specify which generator direction, AB or BA, to use."
        assert 'G' in self.networks.keys()

        with torch.no_grad():
            inverse = direction == 'BA'
            return self.networks['G'](input, inverse=inverse)
