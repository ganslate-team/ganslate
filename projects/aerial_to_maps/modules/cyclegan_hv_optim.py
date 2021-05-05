import torch

from midaGAN.nn.gans.unpaired import cyclegan
from midaGAN import configs
from dataclasses import dataclass
import numpy as np

@dataclass
class AdaptiveTrainingConfig:
    """Adaptive Training config that can be used for many different kinds of 
    adaptive procedures"""
    adaptive_lambda: bool = True
    slack: float = 1.0



@dataclass
class HVOptimizedCycleGANConfig(cyclegan.CycleGANConfig):
    """CycleGAN Config"""
    name: str = "HVOptimizedCycleGAN"
    adaptive_training: AdaptiveTrainingConfig = AdaptiveTrainingConfig()


class HVOptimizedCycleGAN(cyclegan.CycleGAN):
    def __init__(self, conf):
        super().__init__(conf)
        self.nadir_slack = self.conf[self.conf.mode].gan.adaptive_training.slack

    def backward_G(self):
        """Calculate the loss for generators G_AB and G_BA using all specified losses"""

        fake_B = self.visuals['fake_B']  # G_AB(A)
        fake_A = self.visuals['fake_A']  # G_BA(B)

        # ------------------------- GAN Loss ----------------------------
        pred_B = self.networks['D_B'](fake_B)  # D_B(G_AB(A))
        pred_A = self.networks['D_A'](fake_A)  # D_A(G_BA(B))

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
        # Reference for HV optim: https://github.com/belaalb/h-GAN
        adversarial_losses = [self.losses['G_AB'], self.losses['G_BA']]
        cycle_consistency_losses = losses_G.values()
        losses_list = [*adversarial_losses, *cycle_consistency_losses]

        self.update_nadir_point(losses_list)

        combined_loss_G = 0
        for loss in losses_list:
            combined_loss_G -= torch.log(self.nadir_point - loss) 

        self.backward(loss=combined_loss_G, optimizer=self.optimizers['G'], loss_id=0)

    def update_nadir_point(self, losses_list):
        self.nadir_point = torch.tensor(np.max(losses_list) + self.nadir_slack)