import torch
from torch import nn


class PatchNCELoss(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.batch_size = conf.train.batch_size
        self.nce_T = conf.train.gan.optimizer.nce_T

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
