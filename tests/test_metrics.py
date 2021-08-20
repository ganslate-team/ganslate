import sys
from loguru import logger

# --------- ganslate imports ----------
try:
    import ganslate
except ImportError:
    logger.warning("ganslate not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import ganslate

from ganslate.utils.metrics.val_test_metrics import ValTestMetrics
import torch
import numpy as np


def main(conf):
    metricizer = ValTestMetrics(conf)

    # Target array is 10x100x100 out of which 10x50x50 is of value 1000
    target = torch.zeros((10, 100, 100))
    target[:, 0:50, 0:50] = 1000

    # Input array is 10x100x100 of value 500
    input = torch.full_like(target, 500)

    mask = torch.zeros((10, 100, 100), dtype=np.bool)
    mask[:, 0:50, 0:50] = 1

    # If a mask is applied over the 10x50x50 then the MAE should be [(1000 - 500) * size] / size = 500.0
    # but this is not the case if masked_array is not used as the zero values are also considered during the mean.
    print("Metrics without element mask", metricizer.get_metrics(input * mask, target * mask))
    print("Metrics with mask", metricizer.get_metrics(input, target, mask=mask))


if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = OmegaConf.create({
        "mode": "val",
        "val": {
            "metrics": {
                "ssim": True,
                "mae": True,
                "mse": True,
                "psnr": True,
                "nmse": True
            }
        }
    })
    main(conf)
