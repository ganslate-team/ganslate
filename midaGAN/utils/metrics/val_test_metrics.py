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
    input = input.detach().cpu().numpy()
    return input


def create_masked_array(input, mask):
    """
    Create a masked array after applying the respective mask. 
    This mask array will filter values across different operations such as mean
    """
    mask = mask.detach().cpu().numpy()

    mask = mask.astype(np.bool)
    # Masked array needs negated masks as it decides
    # what element to ignore based on True values
    negated_mask = ~mask
    return np.ma.masked_array(input * mask, mask=negated_mask)


# Metrics below are taken from
# https://github.com/facebookresearch/fastMRI/blob/master/fastmri/evaluate.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Added MAE to the list of metrics


def mae(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)"""
    mae_value = np.mean(np.abs(gt - pred))
    return float(mae_value)

def mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE)"""
    mse_value = np.mean((gt - pred)**2)
    return float(mse_value)

def nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Normalized Mean Squared Error (NMSE)"""
    nmse_value = np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2
    return float(nmse_value)

def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    psnr_value = peak_signal_noise_ratio(gt, pred, data_range=gt.max())
    return float(psnr_value)

def ssim(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> float:
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval

    ssim_sum = 0
    size = (gt.shape[0] * gt.shape[1]) if gt.ndim == 4 else gt.shape[0]

    for channel in range(gt.shape[0]):
        # Format is CxHxW or DxHxW
        if gt.ndim == 3:
            ssim_sum = ssim_sum + structural_similarity(gt[channel], pred[channel], data_range=maxval)

        # Format is CxDxHxW
        elif gt.ndim == 4:
            for slice_num in range(gt.shape[1]):
                ssim_sum = ssim_sum + structural_similarity(
                    gt[channel, slice_num], pred[channel, slice_num], data_range=maxval)
        else:
            raise NotImplementedError(f"SSIM for {gt.ndim} images not implemented")

    return ssim_sum / size


METRIC_DICT = {"ssim": ssim, "mse": mse, "nmse": nmse, "psnr": psnr, "mae": mae}


class ValTestMetrics:

    def __init__(self, conf):
        self.conf = conf

    def get_metrics(self, batched_input, batched_target, mask=None):
        batched_input, batched_target = get_npy(batched_input), get_npy(batched_target)
        batched_mask = mask

        metrics = {}

        # Iterating over all metrics that need to be computed
        for metric_name, metric_fn in METRIC_DICT.items():
            if getattr(self.conf[self.conf.mode].metrics, metric_name):
                metric_scores = []

                # If batched mask exists then apply the mask to the inputs and targets
                if batched_mask is not None:
                    batched_input = [create_masked_array(input, mask) \
                                            for input, mask in zip(batched_input, batched_mask)]
                    batched_target = [create_masked_array(target, mask) \
                                            for target, mask in zip(batched_target, batched_mask)]

                # Iterate over input and target batches and compute metrics
                for input, target in zip(batched_input, batched_target):
                    metric_scores.append(metric_fn(target, input))

                # Aggregate metrics over a batch
                metrics[metric_name] = metric_scores

        return metrics

    def get_cycle_metrics(self, batched_input, batched_target):
        batched_input, batched_target = get_npy(batched_input), get_npy(batched_target)

        metrics = {}
        metrics["cycle_SSIM"] = [ssim(target, input) \
                                    for input, target in zip(batched_input, batched_target)]

        return metrics
