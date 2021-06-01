import torch
from midaGAN.nn.losses import cyclegan_losses
from projects.maastro_lung_proton_cbct_to_ct.modules.losses import loss_ops
import monai
# from projects.maastro_lung_proton_cbct_to_ct.modules.losses import registration_loss

LOSS_CRITERION = {
    "Frequency-L1": torch.nn.L1Loss(),
    "Frequency-L2": torch.nn.MSELoss(),
    "Registration": monai.losses.LocalNormalizedCrossCorrelationLoss()

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



class StructureLoss:
    """
    A holder class for different type of structure losses - a structure loss enforces mappings between
    input and translated image to have certain properties.
    """
    def __init__(self, lambda_structure, structure_criterion):
        self.lambda_structure = lambda_structure
        self.type = structure_criterion.split("-")[0]
        self.criterion = LOSS_CRITERION[structure_criterion]

    def __call__(self, input, target):
        self.device = input.device

        if self.type == "Frequency":
            input = loss_ops.get_freq_transform(input)
            target = loss_ops.get_freq_transform(target)
    
        loss = self.criterion(input, target)

        if self.type == "Registration":
            loss = torch.tanh(loss)
            loss = (loss + 1)/2

        return self.lambda_structure * loss

        return self.lambda_structure * loss
