from dataclasses import dataclass

import torch
from midaGAN.nn.gans.unpaired import cyclegan
from midaGAN.nn.losses.adversarial_loss import AdversarialLoss
from modules.losses import structure_loss


@dataclass
class CycleGANwStructureOptimizerConfig(cyclegan.OptimizerConfig):
    """ Structure consistency config for CycleGAN multimodal v1 """
    lambda_structure: float = 0
    structure_criterion: str = "L1"

@dataclass
class CycleGANwStructureConfig(cyclegan.CycleGANConfig):
    name: str = "CycleGANwStructure"
    optimizer: CycleGANwStructureOptimizerConfig = CycleGANwStructureOptimizerConfig()



class CycleGANwStructure(cyclegan.CycleGAN):
    def __init__(self, conf):
        super().__init__(conf)

        # Additional losses used by the model
        structure_loss_names = ['structure_AB', 'structure_BA']
        self.losses.update({name: None for name in structure_loss_names})


    def init_criterions(self):
        # Standard GAN loss
        self.criterion_adv = AdversarialLoss(
            self.conf.train.gan.optimizer.adversarial_loss_type).to(self.device)
        # G losses - Includes Structure-consistency loss
        self.criterion_G = structure_loss.CycleGANLossesWithStructure(self.conf)


