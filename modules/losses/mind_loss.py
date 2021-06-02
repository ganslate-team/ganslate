import math
import torch

class MINDLoss(torch.nn.Module):
    def __init__(self, non_local_region_size=9):
        super().__init__()
        self.get_mind_descriptor = MINDDescriptor()
        self.size = non_local_region_size


    def forward(self, input, target):
        self.get_mind_descriptor.to(input.device)

        # Convert NxCxDxHxW to Nx1xHxW where N = NxD
        input = input.view(-1, 1, *input.shape[3:])
        target = target.view(-1, 1, *target.shape[3:])

        input_features = self.get_mind_descriptor(input)
        target_features = self.get_mind_descriptor(target)

        mind_diff = input_features - target_features
        l1 = torch.norm(mind_diff, 1)
        return l1/(input.shape[2] *input.shape[3] * self.size *self.size)


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

        # Dx1xHxW
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