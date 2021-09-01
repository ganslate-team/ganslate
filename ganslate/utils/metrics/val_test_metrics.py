# import ganslate.nn.losses.ssim as ssim
import numpy as np
from typing import Optional
from scipy.stats import entropy
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

    size = (gt.shape[0] * gt.shape[1]) if gt.ndim == 4 else gt.shape[0]

    ssim_sum = 0
    for channel in range(gt.shape[0]):
        # Format is CxHxW or DxHxW
        if gt.ndim == 3:
            target = gt[channel]
            prediction = pred[channel]
            ssim_sum += structural_similarity(target, prediction, data_range=maxval)

        # Format is CxDxHxW
        elif gt.ndim == 4:
            for slice_num in range(gt.shape[1]):
                target = gt[channel, slice_num]
                prediction = pred[channel, slice_num]
                ssim_sum += structural_similarity(target, prediction, data_range=maxval)
        else:
            raise NotImplementedError(f"SSIM for {gt.ndim} images not implemented")

    return ssim_sum / size


def nmi(gt: np.ndarray, pred: np.ndarray) -> float:
    """Normalized Mutual Information.
    Implementation taken from scikit-image 0.19.0.dev0 source --
        https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/simple_metrics.py#L193-L261
    Not using scikit-image because NMI is supported only in >=0.19.
    """
    bins = 100  # 100 bins by default 
    hist, bin_edges = np.histogramdd(
            [np.reshape(gt, -1), np.reshape(pred, -1)],
            bins=bins,
            density=True,
            )
    H0 = entropy(np.sum(hist, axis=0))
    H1 = entropy(np.sum(hist, axis=1))
    H01 = entropy(np.reshape(hist, -1))
    nmi_value = (H0 + H1) / H01
    return float(nmi_value)


def histogram_chi2(gt: np.ndarray, pred: np.ndarray) -> float:
    """Chi-squared distance computed between global histograms of the GT and the prediction.
    More about comparing two histograms -- 
        https://stackoverflow.com/questions/6499491/comparing-two-histograms
    """
    bins = 100  # 100 bins by default
    
    # Compute histograms
    gt_histogram, gt_bin_edges = np.histogram(gt, bins=bins)
    pred_histogram, pred_bin_edges = np.histogram(pred, bins=bins)
    
    # Normalize the histograms to convert them into discrete distributions
    gt_histogram = gt_histogram / gt_histogram.sum()
    pred_histogram = pred_histogram / pred_histogram.sum()
    
    # Compute chi-squared distance
    bin_to_bin_distances = (pred_histogram - gt_histogram)**2 / (pred_histogram + gt_histogram)
    # Remove NaN values caused by 0/0 division. Equivalent to manually setting them as 0.
    bin_to_bin_distances = bin_to_bin_distances[np.logical_not(np.isnan(bin_to_bin_distances))]
    chi2_distance_value = np.sum(bin_to_bin_distances)
    return float(chi2_distance_value)


METRIC_DICT = {"ssim": ssim, "mse": mse, "nmse": nmse, "psnr": psnr, "mae": mae, "nmi": nmi, "histogram_chi2": histogram_chi2}


class ValTestMetrics:

    def __init__(self, conf):
        self.conf = conf

    def get_metrics(self, inputs, targets, mask=None): 
        inputs, targets = get_npy(inputs), get_npy(targets)
        metrics = {}
        
        # Iterating over all metrics that need to be computed
        for metric_name, metric_fn in METRIC_DICT.items():
            if getattr(self.conf[self.conf.mode].metrics, metric_name):
                metric_scores = []

                # If mask is given, apply it to the inputs and targets
                if mask is not None:
                    inputs = [create_masked_array(i, m) for i, m in zip(inputs, mask)]
                    targets = [create_masked_array(t, m) for t, m in zip(targets, mask)]

                # Iterate over input and target batches and compute metrics
                for input, target in zip(inputs, targets):
                    metric_scores.append(metric_fn(target, input))

                # Aggregate metrics over a batch
                metrics[metric_name] = metric_scores

        return metrics

    def get_cycle_metrics(self, inputs, targets):
        inputs, targets = get_npy(inputs), get_npy(targets)

        metrics = {}
        metrics["cycle_SSIM"] = [ssim(t, i) for i, t in zip(inputs, targets)]

        return metrics
