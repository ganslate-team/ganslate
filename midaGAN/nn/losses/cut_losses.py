import torch
from torch import nn

from midaGAN.nn.losses.utils import ssim
from midaGAN.nn.utils import reshape_to_4D_if_5D

class PatchNCELoss(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.batch_size = conf.batch_size
        self.nce_T = conf.gan.optimizer.nce_T

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, feat_q, feat_k):
        bs, dim = feat_q.shape[:2]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(bs, 1, -1), feat_k.view(bs, -1, 1))
        l_pos = l_pos.view(bs, 1)

        # neg logit
        batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)

        num_patches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(num_patches, device=feat_q.device, dtype=torch.bool)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, num_patches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(
            out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))

        return loss


class PixelIdentityLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.criterion = self._init_criterion(conf)

    def _init_criterion(self, conf):
        self.mode = conf.gan.optimizer.pixel_idt_mode

        # SSIM
        if self.mode == 'ssim':
            channels_ssim = conf.gan.generator.in_channels
            if 'patch_size' in conf.dataset.keys():
                channels_ssim = conf.dataset.patch_size[0]

            return ssim.SSIM(data_range=1,
                             channel=channels_ssim,
                             nonnegative_ssim=True)
        # L1
        elif self.mode == 'l1':
            return nn.L1Loss()

        else:
            raise NotImplementedError(f'Pixel identity loss mode "{self.mode}" not implemented.')

    def forward(self, real, idt):
        # TODO: take care of this in SSIM for both CycleLoss and this loss,
        # make it do this always by abstracting it away
        if self.mode == "ssim":
            # Merge channel and slice dimensions of volume inputs to allow calculation of SSIM
            real = reshape_to_4D_if_5D(real)
            idt = reshape_to_4D_if_5D(idt)

            # Data range needs to be positive and normalized
            # https://github.com/VainF/pytorch-msssim#2-normalized-input
            real = (real + 1) / 2
            idt = (idt + 1) / 2

        return self.criterion(real, idt)
