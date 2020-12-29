import numpy as np
import torch
from torch import nn

from midaGAN.nn.gans.basegan import BaseGAN
from midaGAN.nn.losses.adversarial_loss import AdversarialLoss
from midaGAN.nn.losses.patch_nce import PatchNCELoss
from midaGAN.nn.utils import init_net, get_network_device

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
    mlp_nc: int = 256  # num of features in MLP
    num_patches: int = 256  # number of patches per layer
    use_flip_equivariance: bool = False  # Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT
    optimizer: OptimizerConfig = OptimizerConfig


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
        self.visual_names = {'A': ['real_A', 'fake_B'], 'B': ['real_B', 'idt_B']}
        # get all the names from the above lists into a single flat list
        all_visual_names = [name for v in self.visual_names.values() for name in v]
        # initialize the visuals as None
        self.visuals = {name: None for name in all_visual_names}

        # Losses used by the model
        loss_names = ['D', 'G', 'NCE', 'NCE_idt']
        self.losses = {name: None for name in loss_names}

        # Generators and Discriminators
        network_names = ['G', 'D', 'mlp'] if self.is_train else ['G']  # during test time, only G
        self.networks = {name: None for name in network_names}

        self.setup()  # schedulers, mixed precision, checkpoint loading and network parallelization

    def init_networks(self):
        """Extend the `init_networks` of the BaseGAN by adding the initialization of MLP."""
        super().init_networks()
        if self.is_train:
            channels_per_feature_level = probe_network_channels(self.networks['G'], self.nce_layers,
                                                                self.conf.gan.generator.in_channels)
            mlp = FeaturePatchMLP(channels_per_feature_level, self.conf.gan.num_patches,
                                  self.conf.gan.mlp_nc)
            self.networks['mlp'] = init_net(mlp, self.conf, self.device)

    def init_optimizers(self):
        lr_G = self.conf.gan.optimizer.lr_G
        lr_D = self.conf.gan.optimizer.lr_D
        beta1 = self.conf.gan.optimizer.beta1
        beta2 = self.conf.gan.optimizer.beta2

        self.optimizers['G'] = torch.optim.Adam(self.networks['G'].parameters(),
                                                lr=lr_G,
                                                betas=(beta1, beta2))
        self.optimizers['D'] = torch.optim.Adam(self.networks['D'].parameters(),
                                                lr=lr_D,
                                                betas=(beta1, beta2))
        self.optimizers['mlp'] = torch.optim.Adam(self.networks['mlp'].parameters(),
                                                  lr=lr_G,
                                                  betas=(beta1, beta2))

    def init_criterions(self):
        self.criterion_adv = AdversarialLoss(self.conf.gan.optimizer.adversarial_loss_type).to(
            self.device)
        self.criterion_nce = [
            PatchNCELoss(self.conf).to(self.device) for _ in self.conf.gan.nce_layers
        ]
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
        self.optimizers['mlp'].zero_grad()

        self.backward_G_and_mlp()
        self.optimizers['G'].step()
        self.optimizers['mlp'].step()
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

        loss_real = self.criterion_adv(pred_real, True).mean()
        loss_fake = self.criterion_adv(pred_fake, False).mean()
        self.losses['D'] = loss_real + loss_fake

        self.backward(loss=self.losses['D'], optimizer=self.optimizers['D'], loss_id=0)

    def backward_G_and_mlp(self):
        real_A = self.visuals['real_A']
        real_B = self.visuals['real_B']
        fake_B = self.visuals['fake_B']
        idt_B = self.visuals['idt_B']

        # ------------------------- GAN Loss ----------------------------
        adversarial_loss = 0
        if self.lambda_adv > 0:
            pred_fake = self.networks['D'](fake_B)
            adversarial_loss = self.criterion_adv(pred_fake, True).mean() * self.lambda_adv
            self.losses['G'] = adversarial_loss
        # ---------------------------------------------------------------

        # ------------------------- NCE Loss ----------------------------
        nce_loss = 0
        if self.lambda_nce > 0:
            nce_loss = self.calculate_nce_loss(real_A, fake_B)
            self.losses['NCE'] = nce_loss

            if self.nce_idt:
                nce_idt_loss = self.calculate_nce_loss(real_B, idt_B)
                self.losses['NCE_idt'] = nce_idt_loss
                nce_loss = 0.5 * (nce_loss + nce_idt_loss)
        # ---------------------------------------------------------------

        combined_loss = nce_loss + adversarial_loss

        optimizers = (self.optimizers['G'], self.optimizers['mlp'])
        self.backward(loss=combined_loss, optimizer=optimizers, loss_id=1)

    def calculate_nce_loss(self, source, target):
        source_feats = extract_features(input=source,
                                        network=self.networks['G'],
                                        layers_to_extract_from=self.nce_layers)

        target_feats = extract_features(input=target,
                                        network=self.networks['G'],
                                        layers_to_extract_from=self.nce_layers)

        if self.is_equivariance_flipped:
            target_feats = [feat.flip(-1) for feat in target_feats]

        source_feats_pool, patch_ids = self.networks['mlp'](source_feats)
        target_feats_pool, _ = self.networks['mlp'](target_feats, patch_ids)

        per_level_iterator = zip(target_feats_pool, source_feats_pool, self.criterion_nce,
                                 self.nce_layers)
        nce_loss = 0
        for target_feat, source_feat, criterion, nce_layer in per_level_iterator:
            loss = criterion(target_feat, source_feat) * self.lambda_nce
            nce_loss = nce_loss + loss.mean()
        return nce_loss / len(self.nce_layers)


class FeaturePatchMLP(nn.Module):

    def __init__(self, channels_per_feature, num_patches=256, nc=256):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super().__init__()
        self.num_patches = num_patches
        self.l2norm = LNorm(2)
        self._init_mlp(channels_per_feature, nc)

    def _init_mlp(self, channels_per_feature, mlp_nc):
        self.mlps = nn.ModuleList()

        for input_nc in channels_per_feature:
            mlp = nn.Sequential(nn.Linear(input_nc, mlp_nc), nn.ReLU(), nn.Linear(mlp_nc, mlp_nc))
            self.mlps.append(mlp)

    def forward(self, feats, patch_ids=None):
        device = feats[0].device
        return_feats = []
        return_ids = []

        for i, feat in enumerate(feats):
            # If 3D data (BxCxDxHxW), otherwise 2D (BxCxHxW)
            if len(feat.shape) == 5:
                # Permute and flatten to B, F, C, where F is the flattened D, H and W
                feat = feat.permute(0, 2, 3, 4, 1).flatten(1, 3)
            else:
                # Permute and flatten to B, F, C, where F is the flattened H and W
                feat = feat.permute(0, 2, 3, 1).flatten(1, 2)

            if self.num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[i]
                else:
                    # Randomized indices of the F dimension for selecting patches. If F is 512, it
                    # will be a list with length 512 and will look like, e.g., [511, 3, 275, 303, ...]
                    patch_id = torch.randperm(feat.shape[1], device=device)
                    # Limit the number of patches to `num_patches` if necessary
                    patch_id = patch_id[:int(min(self.num_patches, len(patch_id)))]
                # Select the patches from the feature space
                feat_patch = feat[:, patch_id, :]
            else:
                feat_patch = feat
                patch_id = []

            # Flatten B and F dimensions of the (B, F, C) tensor
            feat_patch = feat_patch.flatten(0, 1)
            feat_patch = self.mlps[i](feat_patch)
            feat_patch = self.l2norm(feat_patch)

            return_feats.append(feat_patch)
            return_ids.append(patch_id)

        return return_feats, return_ids


class LNorm(nn.Module):

    def __init__(self, power=2):
        super().__init__()
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
        f"The encoder has {len(network.encoder)} layers, cannot extract features from layers that do not exist."

    features = []
    feat = input

    for i, layer in enumerate(network.encoder):
        feat = layer(feat)
        if i in layers_to_extract_from:
            features.append(feat)

    return features


def probe_network_channels(network, layers_of_interest, input_channels=3):
    assert len(network.encoder) >= max(layers_of_interest), \
        f"The encoder has {len(network.encoder)} layers, cannot extract features from layers that do not exist."

    channels_per_layer = []
    device = get_network_device(network)

    with torch.no_grad():
        if '3d' in str(network):
            feat = torch.Tensor(1, input_channels, 64, 256, 256).to(device)
        else:
            feat = torch.Tensor(1, input_channels, 256, 256).to(device)

        for i, layer in enumerate(network.encoder):
            feat = layer(feat)
            if i in layers_of_interest:
                channels_per_layer.append(feat.shape[1])

        return channels_per_layer
