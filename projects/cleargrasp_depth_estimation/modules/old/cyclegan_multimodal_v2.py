from dataclasses import dataclass

import torch

from ganslate.nn.gans.unpaired import cyclegan
from ganslate.nn.losses.adversarial_loss import AdversarialLoss

from projects.cleargrasp_depth_estimation.modules.cyclegan_losses_with_structure import CycleGANLossesWithStructure


@dataclass
class OptimizerV2Config(cyclegan.OptimizerConfig):
    """ Optimizer Config CycleGAN multimodal v2 """
    lambda_structure: float = 0


@dataclass
class CycleGANMultiModalV2Config(cyclegan.CycleGANConfig):
    """ CycleGANMultiModalV2 Config """
    _target_: str = "CycleGANMultiModalV2"
    optimizer: OptimizerV2Config = OptimizerV2Config()


class CycleGANMultiModalV2(cyclegan.CycleGAN):
    """ CycleGAN for multimodal images -- Version 2 """

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
        self.criterion_G = CycleGANLossesWithStructure(self.conf, cyclegan_design_version='v2')
