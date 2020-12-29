# coding=utf-8
# Copyright (c) DIRECT Contributors

# Taken from: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
# Licensed under MIT.
# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
# Some changes are made to work together with DIRECT.

# ----------------------------------------------------
# Taken from DIRECT https://github.com/directgroup/direct
# Added support for mixed precision by allowing one image to be of type `half` and the other `float`.

import torch
import torch.nn.functional as F

__all__ = ('batch_ssim', 'ms_ssim', 'SSIM', 'MS_SSIM', 'ThreeComponentSSIM')


def are_tensors_half_or_float(*args):
    for tensor in args:
        if not isinstance(tensor, torch.cuda.FloatTensor) and not isinstance(
                tensor, torch.cuda.HalfTensor):
            return False
    return True


def gradient_image(input):
    input = input.permute(1, 0, 2, 3)

    kernel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    kernel_x = kernel_x.view((1, 1, 3, 3))
    G_x = F.conv2d(input, kernel_x)

    kernel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    kernel_y = kernel_y.view((1, 1, 3, 3))
    G_y = F.conv2d(input, kernel_y)

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
    return G.permute(1, 0, 2, 3)


def _fspecial_gauss_1d(size, sigma):
    """
    Create a 1D gaussian kernel

    Parameters
    ----------
    size : int
        The size of the gaussian kernel
    sigma : float
        The standard deviation of the normal distribution

    Returns
    -------
    torch.Tensor: 1D kernel (1 x 1 x size)

    """
    coords = torch.arange(size).float()
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    """
    Blur input with 1D kernel

    Parameters
    ----------
    input : torch.Tensor
        A batch of tensors to be blurred
    win : torch.Tensor
        1D gaussian kernel

    Returns
    -------
    torch.Tensor: blurred tensors
    """
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
    """
    Calculate the SSIM for input and target.

    Parameters
    ----------
    X : torch.Tensor
    Y : torch.Tensor
    data_range : float or int (optional)
        Value range of input images (typically 1.0 or 255)
    win : 1D gaussian kernel
    size_average :
    K :

    Returns
    -------
    torch.Tensor: SSIM results
    """

    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    S1 = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    S2 = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1

    # SSIM Distance metric approximation from: https://ece.uwaterloo.ca/~z70wang/publications/TIP_SSIM_MathProperties.pdf
    # Clamp to 2 because the max value can only be 2. Due to floating point errors it might sometimes cross it.
    S = torch.clamp(S1 + S2, max=2)
    D_map = torch.sqrt(2 - S)

    D_per_channel = torch.flatten(D_map, 2).mean(-1)
    cs = torch.flatten(S2, 2).mean(-1)
    return D_per_channel, cs, D_map


def batch_ssim(input,
               target,
               data_range=255,
               win_size=11,
               win_sigma=1.5,
               win=None,
               K=(0.01, 0.03),
               nonnegative_ssim=False,
               reduction='mean'):
    r""" interface of ssim
    Args:
        input (torch.Tensor): a batch of images, (N,C,H,W)
        target (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """

    if len(input.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')

    if not input.type() == target.type() and not are_tensors_half_or_float(input, target):
        raise ValueError('Input images should have the same dtype.')

    if not input.shape == target.shape:
        raise ValueError('Input images should have the same shape.')

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(input.shape[1], 1, 1, 1)

    ssim_per_channel, cs, ssim_map = _ssim(input,
                                           target,
                                           data_range=data_range,
                                           win=win,
                                           size_average=False,
                                           K=K)

    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if reduction == 'mean':
        return ssim_per_channel.mean()
    else:
        return ssim_map


def ms_ssim(X,
            Y,
            data_range=255,
            win_size=11,
            win_sigma=1.5,
            win=None,
            weights=None,
            K=(0.01, 0.03),
            reduction='mean'):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        reduction str: if 'mean', ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')

    if not X.type() == Y.type() and not are_tensors_half_or_float(X, Y):
        raise ValueError('Input images should have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same dimensions.')

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** 4), \
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = (X.shape[2] % 2, X.shape[3] % 2)
            X = F.avg_pool2d(X, kernel_size=2, padding=padding)
            Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim**weights.view(-1, 1, 1), dim=0)

    if reduction == 'mean':
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):

    def __init__(self,
                 data_range=255,
                 win_size=11,
                 win_sigma=1.5,
                 channel=3,
                 K=(0.01, 0.03),
                 nonnegative_ssim=False,
                 reduction='mean'):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super().__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.reduction = reduction
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return batch_ssim(X,
                          Y,
                          data_range=self.data_range,
                          win=self.win,
                          K=self.K,
                          nonnegative_ssim=self.nonnegative_ssim,
                          reduction=self.reduction)


class MS_SSIM(torch.nn.Module):

    def __init__(self,
                 data_range=255,
                 win_size=11,
                 win_sigma=1.5,
                 channel=3,
                 weights=None,
                 K=(0.01, 0.03),
                 reduction='mean'):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """
        super().__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.reduction = reduction
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(X,
                       Y,
                       data_range=self.data_range,
                       win=self.win,
                       weights=self.weights,
                       K=self.K,
                       reduction=self.reduction)


class ThreeComponentSSIM(torch.nn.Module):

    def __init__(self,
                 data_range=255,
                 win_size=11,
                 win_sigma=1.5,
                 channel=3,
                 weights=None,
                 K=(0.01, 0.03),
                 reduction=None,
                 multiscale=False,
                 nonnegative_ssim=False):
        r""" class for 3-Component SSIM
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """
        super().__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.reduction = reduction
        self.data_range = data_range
        self.weights = weights
        self.K = K
        self.multiscale = multiscale
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        """
        :param X: Original Image, order is important for threshold computation
        :param Y: Compared Image

        Computations:
        http://utw10503.utweb.utexas.edu/publications/2009/cl_spie09.pdf
        """
        if self.multiscale:
            SSIM_map = ms_ssim(X,
                               Y,
                               data_range=self.data_range,
                               win=self.win,
                               weights=self.weights,
                               K=self.K,
                               reduction=self.reduction)
        else:
            SSIM_map = batch_ssim(X,
                                  Y,
                                  data_range=self.data_range,
                                  win=self.win,
                                  K=self.K,
                                  nonnegative_ssim=self.nonnegative_ssim,
                                  reduction=self.reduction)

        X_grad = gradient_image(X)
        Y_grad = gradient_image(Y)

        # SSIM_map changes shape due to convolution. Crop the values
        X_grad = X_grad[..., :SSIM_map.shape[-2], :SSIM_map.shape[-1]]
        Y_grad = Y_grad[..., :SSIM_map.shape[-2], :SSIM_map.shape[-1]]

        gmax = X_grad.max()
        th1 = 0.12 * gmax
        th2 = 0.06 * gmax

        edge_mask = (X_grad > th1) | (Y_grad > th1)
        smooth_mask = (X_grad < th2) & (Y_grad <= th1)
        texture_mask = ~(edge_mask | smooth_mask)

        edge_map = edge_mask * SSIM_map
        smooth_map = smooth_mask * SSIM_map
        texture_map = texture_mask * SSIM_map

        return .5 * edge_map[edge_mask != 0].mean() + \
                .25 * smooth_map[smooth_mask != 0].mean() + \
                    .25 * texture_map[texture_mask != 0].mean()


if __name__ == "__main__":
    import SimpleITK as sitk
    import numpy as np
    from torchvision.utils import save_image
    from scipy.ndimage.filters import gaussian_filter as blur
    MIN_B = -150
    MAX_B = 150

    img_a = sitk.ReadImage('/workspace/data/CT_LUNG1/train/LUNG1-001_20190225_CT/CT.nrrd')
    img_b = sitk.ReadImage('/workspace/data/CT_LUNG1/train/LUNG1-001_20190225_CT/CT.nrrd')

    array_a = np.clip(sitk.GetArrayFromImage(img_a), MIN_B, MAX_B)
    array_b = np.clip(sitk.GetArrayFromImage(img_b), MIN_B, MAX_B)

    array_a = (array_a - MIN_B) / (MAX_B - MIN_B)
    array_b = (array_b - MIN_B) / (MAX_B - MIN_B)

    channels_ssim = array_a.shape[0]

    array_b = blur(array_b, sigma=7)

    tensor_a = torch.Tensor(array_a).unsqueeze(dim=0)
    tensor_b = torch.Tensor(array_b).unsqueeze(dim=0)

    # tensor_a = gradient_image(tensor_a)
    # tensor_b = gradient_image(tensor_b)

    print(f"Tensor A max: {tensor_a.max()} min: {tensor_a.min()}")
    print(f"Tensor B max: {tensor_b.max()} min: {tensor_b.min()}")

    ssim = ThreeComponentSSIM(channel=channels_ssim, data_range=1)
    ssim_val = ssim(tensor_a, torch.ones_like(tensor_a))

    tensor_a = tensor_a.permute(1, 0, 2, 3)
    tensor_b = tensor_b.permute(1, 0, 2, 3)

    save_image(tensor_a, "ssim_a.png", padding=10)
    save_image(tensor_b, "ssim_b.png", padding=10)

    print(f"Calculated SSIM Value is : {ssim_val}")
