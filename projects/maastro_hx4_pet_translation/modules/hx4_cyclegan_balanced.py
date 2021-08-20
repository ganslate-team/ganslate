from dataclasses import dataclass

import torch

from ganslate.nn.gans.unpaired import cyclegan
from ganslate.nn.losses.adversarial_loss import AdversarialLoss

from projects.maastro_hx4_pet_translation.modules.hx4_cyclegan_balanced_losses \
    import HX4CycleGANBalancedLosses


@dataclass
class HX4CycleGANBalancedConfig(cyclegan.CycleGANConfig):
    """ HX4CycleGANBalanced Config """
    name: str = "HX4CycleGANBalanced"


class HX4CycleGANBalanced(cyclegan.CycleGAN):
    """ Balanced CycleGAN for HX4-PET synthesis
    Notation:
        A1, A2 -- FDG-PET, pCT
        B1, B2 -- HX4-PET, ldCT        
    """
    
    def __init__(self, conf):
        super().__init__(conf)

    def init_criterions(self):
        # Standard GAN loss
        self.criterion_adv = AdversarialLoss(
            self.conf.train.gan.optimizer.adversarial_loss_type).to(self.device)
        # Generator-related losses -- Cycle-consistency and Identity loss
        self.criterion_G = HX4CycleGANBalancedLosses(self.conf)

    def forward(self):
        """Run forward pass; called by both methods <optimize_parameters> and <test>."""
        real_A = self.visuals['real_A']  # [real FDG-PET, real pCT]
        real_B = self.visuals['real_B']  # [real HX4-PET, real ldCT]

        # Forward cycle G_AB (A to B) 
        fake_B1 = self.networks['G_AB'](real_A)  # Compute [fake HX4-PET]
        real_A2 = real_A[:, 1:]  # Get [real pCT]
        rec_A1 = self.networks['G_BA'](torch.cat([fake_B1, real_A2], dim=1))  # Compute [recon FDG-PET], given [fake HX4-PET, real pCT]

        # Backward cycle G_BA (B to A)
        fake_A1 = self.networks['G_BA'](real_B)  # Compute [fake FDG-PET], given [real HX4-PET, real ldCT]
        real_B2 = real_B[:, 1:]  # Get [real ldCT]
        rec_B1 = self.networks['G_AB'](torch.cat([fake_A1, real_B2], dim=1))  # Compute [recon HX4-PET], given [fake FDG-PET, real ldCT]

        # In self.visuals, fake and recon A's and B's are expected to have 2 channels, because 
        # the real ones have 2 channels. This is because the multimodal channel split is specified 
        # for each domain A and B generally, but not for reals, fakes and recons separately.
        # Hack -- Use dummy zeros arrays to fill up the channels of CT components (i.e. the 2nd channel)
        zeros_dummy = torch.zeros_like(real_A2)
        self.visuals.update({ 
            'fake_B': torch.cat([fake_B1, zeros_dummy], dim=1),
            'rec_A': torch.cat([rec_A1, zeros_dummy], dim=1),
            'fake_A': torch.cat([fake_A1, zeros_dummy], dim=1),
            'rec_B': torch.cat([rec_B1, zeros_dummy], dim=1),
        })

    def backward_D(self, discriminator):
        """Calculate GAN loss for the discriminator"""
        # D_B only evaluates HX4-PET
        if discriminator == 'D_B':
            real = self.visuals['real_B'][:, :1]
            fake = self.visuals['fake_B'][:, :1]
            fake = self.fake_B_pool.query(fake)
            loss_id = 0

        # D_A only evaluates FDG-PET
        elif discriminator == 'D_A':
            real = self.visuals['real_A'][:, :1]
            fake = self.visuals['fake_A'][:, :1]
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
        # Get HX4-PET and FDG-PET
        fake_B1 = self.visuals['fake_B'][:, :1]  # G_AB(A)
        fake_A1 = self.visuals['fake_A'][:, :1]  # G_BA(B)

        # ------------------------- GAN Loss ----------------------------
        pred_B = self.networks['D_B'](fake_B1)  # D_B(G_AB(A))
        pred_A = self.networks['D_A'](fake_A1)  # D_A(G_BA(B))

        # Forward GAN loss D_A(G_AB(A))
        self.losses['G_AB'] = self.criterion_adv(pred_B, target_is_real=True)
        # Backward GAN loss D_B(G_BA(B))
        self.losses['G_BA'] = self.criterion_adv(pred_A, target_is_real=True)
        # ---------------------------------------------------------------

        # ------------- G Losses (Cycle, Identity) ----------------------
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
            fake_B1 = self.networks[f'G_{direction}'](input)  # Compute [fake HX4-PET]
            real_A2 = input[:, 1:]                  # Create zeros dummy array to fill up the 2nd channel
            zeros_dummy = torch.zeros_like(real_A2) #
            return torch.cat([fake_B1, zeros_dummy], dim=1)
