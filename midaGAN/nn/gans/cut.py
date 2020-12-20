import numpy as np
import torch
from torch import nn

from midaGAN.nn.gans.basegan import BaseGAN
from midaGAN.nn.losses.adversarial_loss import AdversarialLoss
from midaGAN.nn.losses.patch_nce import PatchNCELoss
from midaGAN.nn.utils import init_net

# Config imports
from dataclasses import dataclass, field
from typing import Tuple
from omegaconf import MISSING
from midaGAN.conf import BaseGANConfig, BaseOptimizerConfig


@dataclass
class OptimizerConfig(BaseOptimizerConfig):
    lambda_adv: float = 1  # weight for Adversarial loss： Adv(G(X))
    lambda_nce: float = 1  # weight for NCE loss: NCE(G(X), X)
    nce_idt: bool = True  # use NCE loss for identity mapping: NCE(G(Y), Y))
    nce_T: float = 0.07  # temperature for NCE loss

@dataclass
class CUTConfig(BaseGANConfig):
    """Contrastive Unpaired Translation (CUT)"""
    name: str = "CUT"
    nce_layers: Tuple[int] = (0, 4, 8, 12, 16)  # compute NCE loss on which layers
    mlp_nc: int = 256 # TODO: same as num_patches?
    num_patches: int = 256  # number of patches per layer
    use_flip_equivariance: bool = False  # Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT
    optimizer: OptimizerConfig = OptimizerConfig

@dataclass
class FastCUTConfig(CUTConfig):
    """Fast Contrastive Unpaired Translation (FastCUT)"""
    name: str = "FastCUT"
    # FastCUT defaults
    use_flip_equivariance: bool = True
    optimizer: OptimizerConfig = OptimizerConfig(nce_idt=False, lambda_nce=10) 

class CUT(BaseGAN):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020
    """
    def __init__(self, conf):
        super().__init__(conf)

        self.lambda_adv = conf.gan.optimizer.lambda_adv
        self.lambda_nce = conf.gan.optimizer.lambda_nce
        self.nce_idt = conf.gan.optimizer.nce_idt
        self.nce_layers = conf.gan.nce_layers
        self.num_patches = conf.gan.num_patches
        self.use_flip_equivariance = conf.gan.use_flip_equivariance
        self.is_equivariance_flipped = False

        # Inputs and Outputs of the model
        self.visual_names = {
            'A': ['real_A', 'fake_B'], 
            'B': ['real_B', 'idt_B']
        }
        # get all the names from the above lists into a single flat list
        all_visual_names = [name for v in self.visual_names.values() for name in v]
        # initialize the visuals as None
        self.visuals = {name: None for name in all_visual_names}

        # Losses used by the model
        loss_names = ['D', 'G', 'NCE', 'NCE_idt']
        self.losses = {name: None for name in loss_names}

        # Generators and Discriminators
        network_names = ['G', 'D', 'mlp'] if self.is_train else ['G'] # during test time, only G
        self.networks = {name: None for name in network_names}

        self.setup(conf) # schedulers, mixed precision, checkpoint loading and network parallelization
    
    def init_networks(self, conf):
        """Extend the `init_networks` of the BaseGAN by adding the initialization of MLP."""
        super().init_networks(conf)
        if self.is_train:
            mlp = PatchSampleF()
            self.networks['mlp'] = init_net(mlp, conf, self.device)

    def init_optimizers(self, conf):
        lr_G = conf.gan.optimizer.lr_G
        lr_D = conf.gan.optimizer.lr_D
        beta1 = conf.gan.optimizer.beta1
        beta2 = conf.gan.optimizer.beta2   

        self.optimizers['G'] = torch.optim.Adam(self.networks['G'].parameters(), lr=lr_G, betas=(beta1, beta2))
        self.optimizers['D'] = torch.optim.Adam(self.networks['D'].parameters(), lr=lr_D, betas=(beta1, beta2))
        # !!!!!!! self.optimizers['mlp'] = torch.optim.Adam(self.networks['mlp'].parameters(), lr=lr_D, betas=(beta1, beta2))

    def init_criterions(self, conf):
        self.criterion_adv = AdversarialLoss(conf.gan.optimizer.adversarial_loss_type).to(self.device)
        self.criterion_nce = [PatchNCELoss(conf).to(self.device) for _ in conf.gan.nce_layers]        
        if self.nce_idt:
            self.criterion_nce_idt = torch.nn.L1Loss().to(self.device)

    def optimize_parameters(self):
        self.forward()

        # ------------------------ Discriminator --------------------------------------------------
        self.set_requires_grad(self.networks['D'], True)
        self.optimizers['D'].zero_grad()
        self.backward_D()
        self.optimizers['D'].step()
        # ------------------------ Generator and MLP ----------------------------------------------
        self.set_requires_grad(self.networks['D'], False)
        self.optimizers['G'].zero_grad()
        # !!!!!!! self.optimizers['mlp'].zero_grad()

        self.backward_G_and_mlp()
        self.optimizers['G'].step()
        # !!!!!!! self.optimizers['mlp'].step()
        # -----------------------------------------------------------------------------------------

    def set_input(self, input):
        """Unpack input data from the dataloader.
        Parameters:
            input (dict) -- a pair of data samples from domain A and domain B.
        """
        self.visuals['real_A'] = input['A'].to(self.device)
        self.visuals['real_B'] = input['B'].to(self.device)

    def forward(self):
        real_A = self.visuals['real_A']
        if self.nce_idt:
            real_B = self.visuals['real_B']

        self.is_equivariance_flipped = False
        if self.is_train and self.use_flip_equivariance and np.random.random() < 0.5:
            self.is_equivariance_flipped = True

            real_A = real_A.flip(-1)  # flip the last dimension
            if self.nce_idt:
                real_B = real_B.flip(-1)
        
        # concat for joint forward?
        self.visuals['fake_B'] = self.networks['G'](real_A)
        if self.nce_idt:
            self.visuals['idt_B'] = self.networks['G'](real_B)

    def backward_D(self):
        real = self.visuals['real_B']
        fake = self.visuals['fake_B']

        pred_real = self.networks['D'](real)
        pred_fake = self.networks['D'](fake.detach())

        loss_real = self.criterion_adv(pred_real, False).mean()
        loss_fake = self.criterion_adv(pred_fake, False).mean()
        self.losses['D'] = loss_real + loss_fake

        self.backward(loss=self.losses['D'], optimizer=self.optimizers['D'], loss_id=0)

    def backward_G_and_mlp(self):
        real_A = self.visuals['real_A']
        real_B = self.visuals['real_B']
        fake_B = self.visuals['fake_B']
        idt_B = self.visuals['idt_B']

        # ------------------------- GAN Loss ----------------------------
        if self.lambda_adv > 0:
            pred_fake = self.networks['D'](fake_B)
            self.losses['G'] = self.criterion_adv(pred_fake, True).mean() * self.lambda_adv
        # ---------------------------------------------------------------

        # ------------------------- NCE Loss ----------------------------
        if self.lambda_nce > 0:
            nce_loss = self.calculate_nce_loss(real_A, fake_B)
            self.losses['NCE'] = nce_loss
            
            if self.nce_idt:
                nce_idt_loss = self.calculate_nce_loss(real_B, idt_B)
                self.losses['NCE_idt'] = nce_idt_loss
                nce_loss = 0.5 * (nce_loss + nce_idt_loss)
        # ---------------------------------------------------------------
        
        combined_loss = nce_loss + self.losses['G']
        
        # !!!!!!! optimizers = (self.optimizers['G'], self.optimizers['mlp'])
        # !!!!!!! self.backward(loss=combined_loss, optimizer=optimizers, loss_id=1)
    
    def calculate_nce_loss(self, source, target):
        nce_layers = self.nce_layers
        num_patches = self.num_patches
        generator = self.networks['G']
        mlp = self.networks['mlp']

        feat_q = extract_features(input=target, 
                                  network=generator, 
                                  layers_to_extract_from=nce_layers)

        if self.use_flip_equivariance and self.is_equivariance_flipped:
            feat_q = [fq.flip(-1) for fq in feat_q]

        feat_k = extract_features(input=source, 
                                  network=generator, 
                                  layers_to_extract_from=nce_layers)

        feat_k_pool, patch_ids = mlp(feat_k, num_patches, patch_ids=None)
        feat_q_pool, _ = mlp(feat_q, num_patches, patch_ids)

        nce_loss = 0
        per_level_iterator = zip(feat_q_pool, feat_k_pool, self.criterion_nce, nce_layers)

        for f_q, f_k, criterion, nce_layer in per_level_iterator:
            continue #  !!!!!!! 
            # !!!!!!! loss = criterion(f_q, f_k) * self.lambda_nce
            # !!!!!!! nce_loss = nce_loss + loss.mean()
        return nce_loss / len(nce_layers)

class FastCUT(CUT):
    """Necessary to be able to build the model with midaGAN.nn.gans.build_gan() 
    when its specified name is "FastCUT."
    """
    def __init__(self, conf):
        super().__init__(conf)

class PatchSampleF(nn.Module):
    def __init__(self, nc=256):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super().__init__()
        self.l2norm = Normalize(2)
        self.nc = nc
        self.mlp_per_layer = {}

    def init_mlp(self, feats):
        for i, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(nn.Linear(input_nc, self.nc), 
                                nn.ReLU(), 
                                nn.Linear(self.nc, self.nc)).to(torch.device('cuda:0')) #  !!!!!!! 
            self.mlp_per_layer[i] = mlp

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_feats = []
        return_ids = []

        if not self.mlp_per_layer:
            self.init_mlp(feats)

        for i, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[i]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
                
            mlp = self.mlp_per_layer[i]
            x_sample = mlp(x_sample)
            x_sample = self.l2norm(x_sample)
            return_ids.append(patch_id)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


def extract_features(input, network, layers_to_extract_from):
    """Extracts features from specified layers for a given input. Assumes that 
    the given network has an attribute `encoder` with the layers of the encoder
    part of the network."""
    assert len(network.encoder) >= max(layers_to_extract_from), \
        f"The encoder has {len(network.encoder)} layers, \cannot extract features from layers that do not exist."

    features = []
    feat = input

    for i, layer in enumerate(network.encoder):
        feat = layer(feat)
        if i in layers_to_extract_from:
            features.append(feat)

    return features