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
        
        input, target = torch2np(input, target)

        metrics = {}

        if self.conf.metrics.ssim:
            metrics.update({
                'SSIM_A': ssim(target, input)
            })

        if self.conf.metrics.mse:
            metrics.update({
                'MSE_A': mse(target, input)
            })

        if self.conf.metrics.nmse:
            metrics.update({
                'NMSE_A': nmse(target, input)
            })

        if self.conf.metrics.psnr:
            metrics.update({
                'PSNR_A': psnr(target, input)
            })                 

        return metrics


def torch2np(input, target):
    input = reshape_to_4D_if_5D(input)
    input = input.detach().cpu().numpy()
        
    target = reshape_to_4D_if_5D(target)
    target = target.detach().cpu().numpy()
    
    # FastMRI metrics use target, input ordering
    return input, target


# Metrics below are taken from 
# https://github.com/facebookresearch/fastMRI/blob/master/fastmri/evaluate.py
# Copyright (c) Facebook, Inc. and its affiliates.

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)

def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def psnr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())

def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
    ) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]
