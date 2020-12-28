import torch
# import midaGAN.nn.losses.ssim as ssim
from midaGAN.nn.utils import reshape_to_4D_if_5D
import numpy as np
from typing import Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class EvaluationMetrics:

    def __init__(self, conf):
        self.conf = conf

    def get_metric_dict(self, input, target):

        input, target = tensor_to_3D_numpy(input, target)

        metrics = {}

        if self.conf.metrics.ssim:
            metrics['SSIM'] = ssim(target, input)

        if self.conf.metrics.mse:
            metrics['MSE'] = mse(target, input)

        if self.conf.metrics.nmse:
            metrics['NMSE'] = nmse(target, input)

        if self.conf.metrics.psnr:
            metrics['PSNR'] = psnr(target, input)

        return metrics


def tensor_to_3D_numpy(input, target):
    input = input.squeeze()
    input = input.detach().cpu().numpy()

    target = target.squeeze()
    target = target.detach().cpu().numpy()

    return input, target


# Metrics below are taken from
# https://github.com/facebookresearch/fastMRI/blob/master/fastmri/evaluate.py
# Copyright (c) Facebook, Inc. and its affiliates.


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred)**2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2


def psnr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(gt[slice_num], pred[slice_num], data_range=maxval)

    return ssim / gt.shape[0]
