import math
import torch

from ganslate.nn.losses import cyclegan_losses


MIND_DESCRIPTOR_CONFIG = {'non_local_region_size': 9, 'patch_size': 7, 'neighbor_size': 3, 'gaussian_patch_sigma': 2.0}


class CycleGANLossesWithStructure(cyclegan_losses.CycleGANLosses):
    """ Additonal constraint:  Structure-consistency loss """

    def __init__(self, conf, cyclegan_design_version):
        super().__init__(conf)

        lambda_structure = conf.train.gan.optimizer.lambda_structure
        use_cuda = conf.train.cuda
        self.cyclegan_design_version = cyclegan_design_version

        if lambda_structure > 0:
            self.criterion_structure = StructureLoss(lambda_structure, use_cuda, cyclegan_design_version)
        else: 
            self.criterion_structure = None

    def __call__(self, visuals):
        real_A, real_B = visuals['real_A'], visuals['real_B']
        fake_A, fake_B = visuals['fake_A'], visuals['fake_B']

        losses = super().__call__(visuals)
        
        # A2-B2 structure-consistency loss
        if self.criterion_structure is not None:
            # MINDLoss(G_AB(real_A)[Depthmap], real_A[Normalmap])
            losses['structure_AB'] = self.lambda_AB * self.criterion_structure(real_A, fake_B)
            # MINDLoss(G_BA(real_B)[Normalmap], real_B[Depthmap])
            losses['structure_BA'] = self.lambda_BA * self.criterion_structure(real_B, fake_A)

        return losses


class StructureLoss:
    """
    Structure-consistency loss -- Yang et al. (2018) - Unpaired Brain MR-to-CT Synthesis using a Structure-Constrained CycleGAN  
    Applied here in 2 different ways: 
        v1 -- Between A2 component (normalmap) and B (depthmap) 
        v2 -- Between A2 (normalmap )and B2 (depthmap) components 
    """
    def __init__(self, lambda_structure, use_cuda, cyclegan_design_version):
        self.lambda_structure = lambda_structure
        self.cyclegan_design_version = cyclegan_design_version

        self.nl_size = MIND_DESCRIPTOR_CONFIG['non_local_region_size']
        self.mind_descriptor = MINDDescriptor(**MIND_DESCRIPTOR_CONFIG) 
        if use_cuda: 
            self.mind_descriptor = self.mind_descriptor.cuda()

    def __call__(self, input_, fake):
        
        if self.cyclegan_design_version == 'v1':
            # Get depthmap and normalmap from appropriate tensors
            if fake.shape[1] == 1:
                depthmap = fake
                normalmap = input_[:, 3:]
            elif fake.shape[1] == 6:
                depthmap = input_
                normalmap = fake[:, 3:]
            
            # Convert normalmap to single channel grayscale (descriptor requires this)             
            normalmap = normalmap.mean(dim=1).unsqueeze(dim=1) 

            # Extract MIND features
            depthmap_features = self.mind_descriptor(depthmap)
            normalmap_features = self.mind_descriptor(normalmap)
            feature_diff = depthmap_features - normalmap_features
            
        elif self.cyclegan_design_version == 'v2':
            # Take the 2nd modality from each tensor
            input_2nd_mod, fake_2nd_mod = input_[:, 3:], fake[:, 3:] 

            # Convert to single channel grayscale (descriptor requires this)
            input_2nd_mod = input_2nd_mod.mean(dim=1).unsqueeze(dim=1) 
            fake_2nd_mod = fake_2nd_mod.mean(dim=1).unsqueeze(dim=1) 

            # Extract MIND features
            input_features = self.mind_descriptor(input_2nd_mod)
            fake_features = self.mind_descriptor(fake_2nd_mod)
            feature_diff = input_features - fake_features
        
        # Compute loss
        l1_distance = torch.norm(feature_diff, 1)
        loss_structure = l1_distance / (input_.shape[2] * input_.shape[3] * self.nl_size * self.nl_size)
        return loss_structure * self.lambda_structure


class MINDDescriptor(torch.nn.Module):
    """
    Taken from the public repository -- https://github.com/tomosu/MIND-pytorch.
    Minor changes made in style for better readability.
    """
    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0):
        super().__init__()
        self.nl_size = non_local_region_size
        self.p_size = patch_size
        self.n_size = neighbor_size
        self.sigma2 = gaussian_patch_sigma * gaussian_patch_sigma

        # Calculate shifted images in non-local region
        self.image_shifter = torch.nn.Conv2d(in_channels=1, out_channels=self.nl_size * self.nl_size,
                                            kernel_size=(self.nl_size, self.nl_size),
                                            stride=1, padding=((self.nl_size - 1) // 2, (self.nl_size - 1) // 2),
                                            dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.nl_size * self.nl_size):
            t = torch.zeros((1, self.nl_size, self.nl_size))
            t[0, i % self.nl_size, i // self.nl_size] = 1
            self.image_shifter.weight.data[i] = t

        # Patch summation
        self.summation_patcher = torch.nn.Conv2d(in_channels=self.nl_size * self.nl_size, out_channels=self.nl_size * self.nl_size,
                                              kernel_size=(self.p_size, self.p_size),
                                              stride=1, padding=((self.p_size - 1) // 2, (self.p_size - 1) // 2),
                                              dilation=1, groups=self.nl_size * self.nl_size, bias=False, padding_mode='zeros')

        for i in range(self.nl_size * self.nl_size):
            # Gaussian kernel
            t = torch.zeros((1, self.p_size, self.p_size))
            cx = (self.p_size - 1) // 2
            cy = (self.p_size - 1) // 2
            for j in range(self.p_size * self.p_size):
                x = j % self.p_size
                y = j // self.p_size
                d2 = torch.norm(torch.tensor([x - cx, y - cy]).float(), 2)
                t[0, x, y] = math.exp(-d2 / self.sigma2)

            self.summation_patcher.weight.data[i] = t

        # Neighbor images
        self.neighbors = torch.nn.Conv2d(in_channels=1, out_channels=self.n_size * self.n_size,
                                        kernel_size=(self.n_size, self.n_size),
                                        stride=1, padding=((self.n_size - 1) // 2, (self.n_size - 1) // 2),
                                        dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t = torch.zeros((1, self.n_size, self.n_size))
            t[0, i % self.n_size, i // self.n_size] = 1
            self.neighbors.weight.data[i] = t

        # Neighbor patcher
        self.neighbor_summation_patcher = torch.nn.Conv2d(in_channels=self.n_size * self.n_size, out_channels=self.n_size * self.n_size,
                                               kernel_size=(self.p_size, self.p_size),
                                               stride=1, padding=((self.p_size - 1) // 2, (self.p_size - 1) // 2),
                                               dilation=1, groups=self.n_size*self.n_size, bias=False, padding_mode='zeros')

        for i in range(self.n_size * self.n_size):
            t = torch.ones((1, self.p_size, self.p_size))
            self.neighbor_summation_patcher.weight.data[i] = t

    def forward(self, orig):
        assert len(orig.shape) == 4
        assert orig.shape[1] == 1

        # Get original image channel stack
        orig_stack = torch.stack([orig.squeeze(dim=1) for i in range(self.nl_size * self.nl_size)], dim=1)

        # Get shifted images
        shifted = self.image_shifter(orig)

        # Get image diff
        diff_images = shifted - orig_stack

        # L2 norm of image diff
        Dx_alpha = self.summation_patcher(torch.pow(diff_images, 2.0))

        # Calculate neighbors' variance
        neighbor_images = self.neighbor_summation_patcher(self.neighbors(orig))
        Vx = neighbor_images.var(dim=1).unsqueeze(dim=1)

        # Output MIND features
        numerator = torch.exp(- Dx_alpha / (Vx + 1e-8))
        denominator = numerator.sum(dim=1).unsqueeze(dim=1)
        mind_features = numerator / denominator
        return mind_features