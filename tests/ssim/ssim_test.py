"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Changes made:
window is now 3D with size (1, 1, win_size, win_size, win_si`ze)
All convolutions for mean and variance comp. are 3D Conv

"""

import torch
from torch import nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        input_dimension = X.ndim - 2
        if input_dimension == 2:
            conv = F.conv2d
            window_size = (self.win_size, self.win_size)
        elif input_dimension == 3:
            conv = F.conv3d
            window_size = (self.win_size, self.win_size, self.win_size)
        else:
            raise f"Unsupported dim {input_dimension} for provided input"

        if not hasattr(self, "w"):
            self.register_buffer("w", torch.ones(1, 1, *window_size) / self.win_size**2)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range)**2
        C2 = (self.k2 * data_range)**2
        ux = conv(X, self.w)  # typing: ignore
        uy = conv(Y, self.w)  #

        uxx = conv(X * X, self.w)
        uyy = conv(Y * Y, self.w)
        uxy = conv(X * Y, self.w)

        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()


if __name__ == "__main__":
    import SimpleITK as sitk
    import numpy as np
    from torchvision.utils import save_image
    from scipy.ndimage.filters import gaussian_filter as blur
    MIN_B = -1000
    MAX_B = 2000

    img_a = sitk.ReadImage('/repos/Maastro/nki_cervix/train/21403922/CT/0/CT.nrrd')
    img_b = sitk.ReadImage('/repos/Maastro/nki_cervix/train/21403922/CT/0/CT.nrrd')

    array_a = np.clip(sitk.GetArrayFromImage(img_a), MIN_B, MAX_B) - MIN_B
    array_b = np.clip(sitk.GetArrayFromImage(img_b), MIN_B, MAX_B) - MIN_B

    # array_a = (array_a - MIN_B) / (MAX_B - MIN_B)
    # array_b = (array_b - MIN_B) / (MAX_B - MIN_B)

    tensor_a = torch.Tensor(array_a).unsqueeze(dim=0).unsqueeze(dim=0)
    tensor_b = torch.Tensor(array_b).unsqueeze(dim=0).unsqueeze(dim=0)

    print(f"Tensor A max: {tensor_a.max()} min: {tensor_a.min()}")
    print(f"Tensor B max: {tensor_b.max()} min: {tensor_b.min()}")

    ssim = SSIMLoss()
    ssim_val = ssim(tensor_a, tensor_b, data_range=torch.full((1, 1, 1, 1), tensor_a.max()))

    print(f"Calculated SSIM Value is : {ssim_val}")
