# import midaGAN.nn.losses.ssim as ssim
import numpy as np
from typing import Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def get_npy(input):
    """
    Gets numpy array from torch tensor after squeeze op.
    If a mask is provided, a masked array is created.
    """
    input = input.squeeze()
    input = input.detach().cpu().numpy()
    return input

def create_masked_array(input, mask):
    """
    Create a masked array after applying the respective mask. 
    This mask array will filter values across different operations such as mean
    """
    mask = mask.squeeze()
    mask = mask.detach().cpu().numpy()
    mask = mask.astype(np.bool)
    # Masked array needs negated masks as it decides 
    # what element to ignore based on True values
    negated_mask = ~mask
    return np.ma.masked_array(input*mask, mask=negated_mask)

# Metrics below are taken from
# https://github.com/facebookresearch/fastMRI/blob/master/fastmri/evaluate.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Added MAE to the list of metrics

def mae(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Absolute Error (MAE)"""
    return np.mean(np.abs(gt - pred))

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

METRIC_DICT = {
    "ssim": ssim,
    "mse":  mse,
    "nmse": nmse,
    "psnr": psnr,
    "mae":  mae
}

class EvaluationMetrics:
    def __init__(self, conf):
        self.conf = conf

    def get_metrics(self, input, target, mask=None):
        input, target = get_npy(input), get_npy(target)

        # Apply masks if provided
        if mask is not None:
            input = create_masked_array(input, mask)
            target = create_masked_array(target, mask)

        metrics = {}
        for metric_name, metric_fn in METRIC_DICT.items():
            if getattr(self.conf[self.conf.mode].metrics, metric_name):
                metrics[metric_name] = metric_fn(target, input)

        return metrics

    def get_cycle_metrics(self, input, target):
        input = get_npy(input)
        target = get_npy(target)
        metrics = {}
        metrics["cycle_SSIM"] = ssim(input, target)

        return metrics
