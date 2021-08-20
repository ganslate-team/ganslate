from dataclasses import dataclass

import torch

from midaGAN.nn.gans.unpaired import cyclegan
from midaGAN.nn.losses.adversarial_loss import AdversarialLoss

from projects.cleargrasp_depth_estimation.modules.cyclegan_losses_for_v3 \
    import CycleGANLossesForV3


@dataclass
class CycleGANMultiModalV3Config(cyclegan.CycleGANConfig):
    """ CycleGANMultiModalV3 Config """
    name: str = "CycleGANMultiModalV3"


class CycleGANMultiModalV3(cyclegan.CycleGAN):
    """ CycleGAN for multimodal images -- Version 3 
    a.k.a CycleGAN-balanced
    
    Notation:
        A1, A2 -- rgb_A, normalmap
        B1, B2 -- rgb_B, depthmap        
    """
    
    def __init__(self, conf):
        super().__init__(conf)

    def init_criterions(self):
        # Standard GAN loss
        self.criterion_adv = AdversarialLoss(
            self.conf.train.gan.optimizer.adversarial_loss_type).to(self.device)
        # Generator-related losses -- Cycle-consistency and Identity loss
        self.criterion_G = CycleGANLossesForV3(self.conf)

    def forward(self):
        """Run forward pass; called by both methods <optimize_parameters> and <test>."""
        real_A = self.visuals['real_A']
        real_B = self.visuals['real_B']

        # Forward cycle G_AB (A to B) 
        fake_B2 = self.networks['G_AB'](real_A)  # Compute depthmap
        real_A1 = real_A[:, :3]  # Get rgb_A
        rec_A2 = self.networks['G_BA'](torch.cat([real_A1, fake_B2], dim=1))  # Compute normalmap recon.

        # Backward cycle G_BA (B to A)
        fake_A2 = self.networks['G_BA'](real_B)  # Compute normalmap
        real_B1 = real_B[:, :3]  # Get rgb_B
        rec_B2 = self.networks['G_AB'](torch.cat([real_B1, fake_A2], dim=1))  # Compute depthmap recon.

        # Hack -- Use dummy zeros arrays to fill up the channels of rgb components
        dummy_array = torch.zeros_like(real_A1)
        self.visuals.update({ 
            'fake_B': torch.cat([dummy_array, fake_B2], dim=1),
            'rec_A': torch.cat([dummy_array, rec_A2], dim=1),
            'fake_A': torch.cat([dummy_array, fake_A2], dim=1),
            'rec_B': torch.cat([dummy_array, rec_B2], dim=1),
        })

    def backward_D(self, discriminator):
        """Calculate GAN loss for the discriminator"""
        # D_B only evaluates depthmap
        if discriminator == 'D_B':
            real = self.visuals['real_B'][:, 3:]
            fake = self.visuals['fake_B'][:, 3:]
            fake = self.fake_B_pool.query(fake)
            loss_id = 0

        # D_A only evaluates normalmap
        elif discriminator == 'D_A':
            real = self.visuals['real_A'][:, 3:]
            fake = self.visuals['fake_A'][:, 3:]
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
        """Calculate the loss for generators G_AB and G_BA using all specified losses"""
        # Get depthmap and normalmap
        fake_B2 = self.visuals['fake_B'][:, 3:]  # G_AB(A)
        fake_A2 = self.visuals['fake_A'][:, 3:]  # G_BA(B)

        # ------------------------- GAN Loss ----------------------------
        pred_B = self.networks['D_B'](fake_B2)  # D_B(G_AB(A))
        pred_A = self.networks['D_A'](fake_A2)  # D_A(G_BA(B))

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
        self.backward(loss=combined_loss_G, optimizer=self.optimizers['G'], loss_id=0)



    def infer(self, input, direction='AB'):
        assert direction in ['AB', 'BA'], "Specify which generator direction, AB or BA, to use."
        assert f'G_{direction}' in self.networks.keys()

        with torch.no_grad():
            fake_B2 = self.networks[f'G_{direction}'](input)
            real_A1 = input[:, :3]
            return torch.cat([torch.zeros_like(real_A1), fake_B2], dim=1)