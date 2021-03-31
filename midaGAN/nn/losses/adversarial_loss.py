from typing import Union, Dict

import torch
from torch import nn


class AdversarialLoss(nn.Module):
    """Define different GAN objectives.
    The AdversarialLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the AdversarialLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. 
            It currently supports vanilla, lsgan, and wgangp.

            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError(f"GAN mode {gan_mode} not implemented.")

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def calculate_loss(self, prediction: torch.Tensor, target_is_real: bool):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            bs = prediction.size(0)
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss


    def forward(self, prediction: Union[Dict[str, torch.Tensor], torch.Tensor], \
                    target_is_real: bool):
        """
        Calculate loss given Discriminator's output and ground truth labels.
        Parameters:
            prediction (tensor or dictionary of tensors) - the prediction output can be 
            from a single discriminator where it is a tensor or from multiple 
            discriminators or more complex discriminator structures where
            it can be a Dict of tensors. When it is a dict of tensors the adversarial loss
            is averaged (TODO: will be configurable) over all the elements of the dict. 
            target_is_real (bool) - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        # If prediction is a dict, compute loss and reduce over all keys of the dict
        if isinstance(prediction, dict):
            loss_list = [self.calculate_loss(pred, target_is_real) for pred in prediction.values()]
            loss = torch.stack(loss_list).mean()
        else:
            loss = self.calculate_loss(prediction, target_is_real)

        return loss
