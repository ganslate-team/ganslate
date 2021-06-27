import torch
from midaGAN.nn.losses import cyclegan_losses
from modules.losses import frequency_loss
from modules.losses import mind_loss
from modules.losses import registration_loss


LOSS_CRITERION = {
    "Frequency-L1": frequency_loss.FrequencyLoss(distance=torch.nn.L1Loss),
    "Frequency-L2": frequency_loss.FrequencyLoss(distance=torch.nn.MSELoss),
    "Registration": registration_loss.RegistrationLoss(),
    "MIND": mind_loss.MINDLoss()
}

class CycleGANLossesWithStructure(cyclegan_losses.CycleGANLosses):
    """ Additonal constraint:  Structure-consistency loss """
    def __init__(self, conf):
        super().__init__(conf)
        lambda_structure = conf[conf.mode].gan.optimizer.lambda_structure
        structure_criterion = conf[conf.mode].gan.optimizer.structure_criterion

        self.structure_loss = StructureLoss(lambda_structure, structure_criterion)


    def __call__(self, visuals):
        real_A, real_B = visuals['real_A'], visuals['real_B']
        fake_A, fake_B = visuals['fake_A'], visuals['fake_B']

        losses = super().__call__(visuals)
        
        # A2-B2 structure-consistency loss
        if self.structure_loss is not None:
            # MINDLoss(G_AB(real_A)[Depthmap], real_A[Normalmap])
            losses['structure_AB'] = self.lambda_AB * self.structure_loss(real_A, fake_B)
            # MINDLoss(G_BA(real_B)[Normalmap], real_B[Depthmap])
            losses['structure_BA'] = self.lambda_BA * self.structure_loss(real_B, fake_A)

        return losses



class StructureLoss(torch.nn.Module):
    """
    A holder class for different type of structure losses - a structure loss enforces mappings between
    input and translated image to have certain properties.
    """
    def __init__(self, lambda_structure, structure_criterion):
        super().__init__()

        assert len(lambda_structure) == len(structure_criterion), \
            "Length of lambda and structure does not match"

        self.lambda_structure = lambda_structure
        self.criterion = []
        for criterion in structure_criterion:
            self.criterion.append(LOSS_CRITERION[criterion])

    def forward(self, input, target):
        loss = 0
        for _lambda, criterion in zip(self.lambda_structure, self.criterion):
            loss += _lambda * criterion(input, target)
        return loss
