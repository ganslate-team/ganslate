import torch

from midaGAN.nn.losses import cyclegan_losses


class CycleGANLossesWithStructure(cyclegan_losses.CycleGANLosses):
    """ Additonal constraint: Structure-consistency loss """

    def __init__(self, conf):
        super().__init__(conf)

        lambda_structure = conf.train.gan.optimizer.lambda_structure

        if lambda_structure > 0:
            self.criterion_structure = StructureLoss(lambda_structure)
        else: 
            self.criterion_structure = None

    def __call__(self, visuals):
        real_A, real_B = visuals['real_A'], visuals['real_B']
        fake_A, fake_B = visuals['fake_A'], visuals['fake_B']

        losses = super().__call__(visuals)
        
        # A1-B1 structure-consistency loss
        if self.criterion_structure is not None:
            # || G_AB(real_A)[RGB] - real_A[RGB] ||
            losses['structure_AB'] = self.lambda_AB * self.criterion_structure(real_A, fake_B)
            # || G_BA(real_B)[RGB] - real_B[RGB] ||
            losses['structure_BA'] = self.lambda_BA * self.criterion_structure(real_B, fake_A)

        return losses


class StructureLoss:
    """
    Structure-consistency loss -- Yang et al. (2018) - Unpaired Brain MR-to-CT Synthesis using a Structure-Constrained CycleGAN  
    Using L1 now for simplicity.  TODO: Change to MIND loss (https://github.com/tomosu/MIND-pytorch)
    Applied here to constrain the A1 and B1 components (here, RGB photos) of multi-modal A and B to have same content
    """
    def __init__(self, lambda_structure):
        self.lambda_structure = lambda_structure
        self.criterion = torch.nn.L1Loss()

    def __call__(self, input_, translated):
        loss_structure = self.criterion(input_[:, :3], translated[:, :3])  
        return loss_structure * self.lambda_structure
