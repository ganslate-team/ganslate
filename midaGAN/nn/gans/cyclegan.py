import itertools
from dataclasses import dataclass

import torch
from midaGAN import configs
from midaGAN.data.utils.image_pool import ImagePool
from midaGAN.nn.gans.basegan import BaseGAN
from midaGAN.nn.losses.adversarial_loss import AdversarialLoss
from midaGAN.nn.losses.cyclegan_losses import CycleGANLosses


@dataclass
class OptimizerConfig(configs.base.BaseOptimizerConfig):
    lambda_A: float = 10.0
    lambda_B: float = 10.0
    lambda_identity: float = 0
    lambda_inverse: float = 0
    proportion_ssim: float = 0.84
    ssim_type: str = "SSIM"  # Possible options are ThreeComponentSSIM, SSIM, MS-SSIM


@dataclass
class CycleGANConfig(configs.base.BaseGANConfig):
    """CycleGAN"""
    name: str = "CycleGAN"
    pool_size: int = 50
    optimizer: OptimizerConfig = OptimizerConfig


class CycleGAN(BaseGAN):
    """CycleGAN"""

    def __init__(self, conf):
        super().__init__(conf)

        # Inputs and Outputs of the model
        self.visual_names = {
            'A': ['real_A', 'fake_B', 'rec_A', 'idt_A'],
            'B': ['real_B', 'fake_A', 'rec_B', 'idt_B']
        }
        # get all the names from the above lists into a single flat list
        all_visual_names = [name for v in self.visual_names.values() for name in v]
        # initialize the visuals as None
        self.visuals = {name: None for name in all_visual_names}

        # Losses used by the model
        loss_names = [
            'D_A', 'G_A', 'cycle_A', 'idt_A', 'inv_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'inv_B'
        ]
        self.losses = {name: None for name in loss_names}

        # Optimizers
        optimizer_names = ['G', 'D']
        self.optimizers = {name: None for name in optimizer_names}

        # Generators and Discriminators
        network_names = ['G_A', 'G_B', 'D_A', 'D_B'] if self.is_train else ['G_A']
        self.networks = {name: None for name in network_names}

        if self.is_train:
            # Create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(conf.train.gan.pool_size)
            self.fake_B_pool = ImagePool(conf.train.gan.pool_size)

        # Set up networks, optimizers, schedulers, mixed precision, checkpoint loading, network parallelization...
        self.setup()

    def init_criterions(self):
        # Standard GAN loss
        self.criterion_adv = AdversarialLoss(self.conf.train.gan.optimizer.adversarial_loss_type).to(
            self.device)
        # Generator-related losses -- Cycle-consistency, Identity and Inverse loss
        self.criterion_G = CycleGANLosses(self.conf)

    def init_optimizers(self):
        lr_G = self.conf.train.gan.optimizer.lr_G
        lr_D = self.conf.train.gan.optimizer.lr_D
        beta1 = self.conf.train.gan.optimizer.beta1
        beta2 = self.conf.train.gan.optimizer.beta2

        params_G = itertools.chain(self.networks['G_A'].parameters(),
                                   self.networks['G_B'].parameters())
        params_D = itertools.chain(self.networks['D_A'].parameters(),
                                   self.networks['D_B'].parameters())

        self.optimizers['G'] = torch.optim.Adam(params_G, lr=lr_G, betas=(beta1, beta2))
        self.optimizers['D'] = torch.optim.Adam(params_D, lr=lr_D, betas=(beta1, beta2))

        self.setup_loss_masking(self.conf.train.gan.optimizer.loss_mask)

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
        discriminators = [self.networks['D_A'], self.networks['D_B']]

        self.forward()  # compute fake images and reconstruction images.

        # Compute generator based metrics dependent on visuals
        self.metrics.update(self.training_metrics.compute_metrics_G(self.visuals))

        # Mask visuals if masking for certain value enabled
        self.mask_current_visuals()

        # ------------------------ G (A and B) ----------------------------------------------------
        self.set_requires_grad(discriminators, False)  # Ds require no gradients when optimizing Gs
        self.optimizers['G'].zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizers['G'].step()  # update G's weights
        # ------------------------ D_A and D_B ----------------------------------------------------
        self.set_requires_grad(discriminators, True)
        self.optimizers['D'].zero_grad()  #set D_A and D_B's gradients to zero
        self.backward_D('D_A')  # calculate gradients for D_A

        # Update metrics for D_A
        self.metrics.update(
            self.training_metrics.compute_metrics_D('D_A', self.pred_real, self.pred_fake))

        self.backward_D('D_B')  # calculate gradients for D_B

        # Update metrics for D_B
        self.metrics.update(
            self.training_metrics.compute_metrics_D('D_B', self.pred_real, self.pred_fake))

        self.optimizers['D'].step()  # update D_A and D_B's weights
        # -----------------------------------------------------------------------------------------

    def forward(self):
        """Run forward pass; called by both methods <optimize_parameters> and <test>."""
        real_A = self.visuals['real_A']
        real_B = self.visuals['real_B']

        # Forward cycle G_A (A to B)
        fake_B = self.networks['G_A'](real_A)
        rec_A = self.networks['G_B'](fake_B)

        # Backward cycle G_B (B to A)
        fake_A = self.networks['G_B'](real_B)  # G forward is G_A, G inverse forward is G_B
        rec_B = self.networks['G_A'](fake_A)

        # Visuals for Identity loss
        idt_A, idt_B = None, None
        if self.criterion_G.is_using_identity():
            idt_A = self.networks['G_B'](real_A)
            idt_B = self.networks['G_A'](real_B)

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
        if discriminator == 'D_A':
            real = self.visuals['real_B']
            fake = self.visuals['fake_B']
            fake = self.fake_B_pool.query(fake)
            loss_id = 0

        elif discriminator == 'D_B':
            real = self.visuals['real_A']
            fake = self.visuals['fake_A']
            fake = self.fake_A_pool.query(fake)
            loss_id = 1
        else:
            raise ValueError('The discriminator has to be either "D_A" or "D_B".')

        self.pred_real = self.networks[discriminator](real)

        # Detaching fake: https://github.com/pytorch/examples/issues/116
        self.pred_fake = self.networks[discriminator](fake.detach())

        loss_real = self.criterion_adv(self.pred_real, target_is_real=True)
        loss_fake = self.criterion_adv(self.pred_fake, target_is_real=False)
        self.losses[discriminator] = loss_real + loss_fake

        # backprop
        self.backward(loss=self.losses[discriminator], optimizer=self.optimizers['D'], loss_id=2)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B using all specified losses"""

        fake_A = self.visuals['fake_A']  # G_B(B)
        fake_B = self.visuals['fake_B']  # G_A(A)

        # ------------------------- GAN Loss ----------------------------
        pred_A = self.networks['D_A'](fake_B)  # D_A(G_A(A))
        pred_B = self.networks['D_B'](fake_A)  # D_B(G_B(B))

        # Forward GAN loss D_A(G_A(A))
        self.losses['G_A'] = self.criterion_adv(pred_A, target_is_real=True)
        # Backward GAN loss D_B(G_B(B))
        self.losses['G_B'] = self.criterion_adv(pred_B, target_is_real=True)
        # ---------------------------------------------------------------

        # ------------- G Losses (Cycle, Identity, Inverse) -------------
        losses_G = self.criterion_G(self.visuals)
        self.losses.update(losses_G)
        # ---------------------------------------------------------------

        # combine losses and calculate gradients
        combined_loss_G = sum(losses_G.values()) + self.losses['G_A'] + self.losses['G_B']
        self.backward(loss=combined_loss_G, optimizer=self.optimizers['G'], loss_id=0)
        
    def infer(self, input, cycle='A'):
        assert cycle in ['A', 'B'], \
            "Infer needs an input of either cycle with A or B domain as input"
        assert f'G_{cycle}' in self.networks.keys()

        with torch.no_grad():
            return self.networks[f'G_{cycle}'].forward(input)
