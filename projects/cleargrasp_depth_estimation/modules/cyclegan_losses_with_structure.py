import torch

from midaGAN.nn.losses import cyclegan_losses


class CycleGANLossesWithStructure(cyclegan_losses.CycleGANLosses):
    """ Additonal constraint: Structure-consistency loss """

    def __init__(self, conf):
        super().__init__(conf)

        lambda_structure = conf.train.gan.optimizer.lambda_structure

        if lambda_structure > 0:
            self.criterion_structure = StructureLoss(lambda_structure, self.lambda_A, self.lambda_B)
        else: 
            self.criterion_structure = None

    def __call__(self, visuals):
        real_A, real_B = visuals['real_A'], visuals['real_B']
        fake_A, fake_B = visuals['fake_A'], visuals['fake_B']

        losses = super().__call__(visuals)
        
        # A1-B1 structure-consistency loss
        if self.criterion_structure is not None:
            losses['structure_AB'], losses['structure_BA'] = self.criterion_structure(real_A, real_B, fake_A, fake_B)

        return losses


class StructureLoss:
    """
    Structure-consistency loss --  Yang et al. (2018) - Unpaired Brain MR-to-CT Synthesis using a Structure-Constrained CycleGAN  
    Using L1 now for simplicity. TODO: Change to MIND loss (https://github.com/tomosu/MIND-pytorch)
    Applied here to constrain the A1 and B1 components (here, RGB photos) of multi-modal A and B to have same content
    """
    def __init__(self, lambda_structure, lambda_A, lambda_B):
        self.lambda_structure = lambda_structure
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.criterion = torch.nn.L1Loss()

    def __call__(self, real_A, real_B, fake_A, fake_B):
        loss_structure_AB = self.criterion(real_A[:, :3], fake_B[:, :3])  # || G_AB(real_A)[RGB] - real_A[RGB] ||
        loss_structure_BA = self.criterion(real_B[:, :3], fake_A[:, :3])  # || G_BA(real_B)[RGB] - real_B[RGB] ||

        loss_structure_AB = loss_structure_AB * self.lambda_A * self.lambda_structure
        loss_structure_BA = loss_structure_BA * self.lambda_B * self.lambda_structure
        return loss_structure_AB, loss_structure_BA
