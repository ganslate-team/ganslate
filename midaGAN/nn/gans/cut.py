import numpy as np
import torch
from torch import nn

from midaGAN.nn.gans.basegan import BaseGAN
from midaGAN.nn.losses.patch_nce import PatchNCELoss

# Config imports
from dataclasses import dataclass, field
from typing import Tuple
from omegaconf import MISSING
from midaGAN.conf import BaseGANConfig, BaseOptimizerConfig


@dataclass
class OptimizerConfig(BaseOptimizerConfig):
    lambda_gan: float = 1  # weight for GAN loss：GAN(G(X))
    lambda_nce: float = 1  # weight for NCE loss: NCE(G(X), X)
    nce_idt: bool = True  # use NCE loss for identity mapping: NCE(G(Y), Y))
    nce_T: float = 0.07  # temperature for NCE loss

@dataclass
class CUTConfig(BaseGANConfig):
    """Contrastive Unpaired Translation (CUT)"""
    name: str = "CUT"
    nce_layers: Tuple[int] = (0, 4, 8, 12, 16)  # compute NCE loss on which layers
    netF: str = 'mlp_sample' # TODO: remove
    netF_nc: int = 256 # TODO: same as num_patches?
    num_patches: int = 256  # number of patches per layer
    flip_equivariance: bool = False  # Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT
    optimizer: OptimizerConfig = OptimizerConfig

@dataclass
class FastCUTConfig(CUTConfig):
    """Fast Contrastive Unpaired Translation (FastCUT)"""
    name: str = "FastCUT"
    # FastCUT defaults
    flip_equivariance: bool = True
    optimizer: OptimizerConfig = OptimizerConfig(nce_idt=False, lambda_nce=10) 

class CUT(BaseGAN):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020
    """
    def __init__(self, conf):
        super().__init__(conf)

        # Inputs and Outputs of the model
        self.visual_names = {
            'A': ['real_A', 'fake_B', 'idt_B'], 
            'B': ['real_B']
        }
        # get all the names from the above lists into a single flat list
        all_visual_names = [name for v in self.visual_names.values() for name in v]
        # initialize the visuals as None
        self.visuals = {name: None for name in all_visual_names}

        # if opt.nce_idt and self.isTrain:
        #     self.loss_names += ['NCE_Y']
        #     self.visual_names += ['idt_B']

        # Losses used by the model
        self.loss_names = ['D', 'G', 'NCE', 'NCE_idt'] # 'G_GAN' 
        self.losses = {name: None for name in loss_names}

        # Generators and Discriminators
        network_names = ['G', 'D', 'mlp'] if self.is_train else ['G'] # during test time, only G
        self.networks = {name: None for name in network_names}

        self.setup() # schedulers, mixed precision, checkpoint loading and network parallelization
    
    def init_networks(self, conf):
        """Extend the `init_networks` of the BaseGAN by adding the initialization of MLP."""
        super().init_networks(conf)
        
        if self.is_train:
            mlp = PatchSampleF(conf)
            self.networks['mlp'] = init_net(conf, mlp, self.device)


    def init_optimizers(self, conf):
        lr_G = conf.gan.optimizer.lr_G
        lr_D = conf.gan.optimizer.lr_D
        beta1 = conf.gan.optimizer.beta1
        beta2 = conf.gan.optimizer.beta2   

        self.optimizers['G'] = torch.optim.Adam(self.networks['G'].parameters(), lr=lr_G, betas=(beta1, beta2))
        self.optimizers['D'] = torch.optim.Adam(self.networks['D'].parameters(), lr=lr_D, betas=(beta1, beta2))
        self.optimizers['mlp'] = torch.optim.Adam(self.networks['mlp'].parameters(), lr=lr_D, betas=(beta1, beta2))                           

        self.setup_loss_masking(conf.gan.optimizer.loss_mask)

    def init_criterions(self, conf):
        # Standard GAN loss 
        self.criterion_advers = AdversarialLoss(conf.gan.optimizer.adversarial_loss_type).to(self.device)
        self.criterion_nce = [PatchNCELoss(conf).to(self.device) for _ in conf.gan.nce_layers]
        self.criterion_idt = torch.nn.L1Loss().to(self.device)

    def optimize_parameters(self):
        self.forward()

        # ------------------------ Discriminator --------------------------------------------------
        self.set_requires_grad(self.networks['D'], True)
        self.optimizer['D'].zero_grad()
        self.backward_D()
        self.optimizer['D'].step()
        # ------------------------ Generator and MLP ----------------------------------------------
        self.set_requires_grad(self.networks['D'], False)
        self.optimizer['G'].zero_grad()
        self.optimizer['mlp'].zero_grad()

        self.backward_G_and_mlp()
        self.optimizer['G'].step()
        self.optimizer['mlp'].step()
        # -----------------------------------------------------------------------------------------

    def set_input(self, input):
        """Unpack input data from the dataloader.
        Parameters:
            input (dict) -- a pair of data samples from domain A and domain B.
        """
        self.visuals['real_A'] = input['A'].to(self.device)
        self.visuals['real_B'] = input['B'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def backward_D():
        real = self.visuals['real_B']
        fake = self.visuals['fake_B']

        pred_real = self.networks['D'](real)
        pred_fake = self.networks['D'](fake.detach())

        loss_real = self.criterion_advers(pred_real, False).mean()
        loss_fake = self.criterion_advers(pred_fake, False).mean()
        self.losses['D'] = loss_real + loss_fake

        self.backward(loss=self.losses['D'], optimizer=self.optimizers['D'], loss_id=0)

    def backward_G_and_mlp():
        real_A = self.visuals['real_A']
        real_B = self.visuals['real_B']
        fake_B = self.visuals['fake_B']
        idt_B = self.visuals['idt_B']

        # ------------------------- GAN Loss ----------------------------
        # if lambda_gan > 0
        pred_fake = self.networks['D'](fake_B)
        self.losses['G'] = self.criterion_advers(pred_fake, True).mean() * self.gan.optimizer.lambda_gan
        # ---------------------------------------------------------------

        # ------------------------- NCE Loss ----------------------------
        # if lambda_nce > 0
        nce_loss = self.calculate_NCE_loss(real_A, fake_B)
        self.losses['NCE'] = nce_loss
        # if above and nce_idt
        nce_idt_loss = self.calculate_NCE_loss(real_B, idt_B)
        self.losses['NCE_idt'] = nce_idt_loss
        # if above and nce_idt
        nce_loss = 0.5 * (nce_loss + nce_idt_loss)
        # ---------------------------------------------------------------

        combined_loss = nce_loss + self.losses['G']
        optimizers = [self.optimizers['G'], self.optimizers['mlp']]
        self.backward(loss=combined_loss, optimizer=optimizers, loss_id=1)

    def calculate_NCE_loss(self, src, tgt):
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / len(self.nce_layers)

class FastCUT(CUT):
    """Necessary to be able to build the mode lwith midaGAN.nn.gans.build_gan() 
    when its specified name is "FastCUT."
    """
    def __init__(self, conf):
        super().__init__(conf)

class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, nc=256, init_type='normal', init_gain=0.02, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super().__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids