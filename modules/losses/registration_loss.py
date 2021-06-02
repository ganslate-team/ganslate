import monai
import torch

class RegistrationLoss(monai.losses.LocalNormalizedCrossCorrelationLoss):
    def forward(self, input, target):
        loss = super().forward(input, target)
        loss = torch.tanh(loss)
        loss = (loss + 1)/2
        return loss