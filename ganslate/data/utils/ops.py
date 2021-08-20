import numpy as np


def pad(volume, target_shape):
    assert len(target_shape) == len(volume.shape)
    # By default no padding
    pad_width = [(0, 0) for _ in range(len(target_shape))]

    for dim in range(len(target_shape)):
        if target_shape[dim] > volume.shape[dim]:
            pad_total = target_shape[dim] - volume.shape[dim]
            pad_per_side = pad_total // 2
            pad_width[dim] = (pad_per_side, pad_total % 2 + pad_per_side)

    return np.pad(volume, pad_width, 'constant', constant_values=volume.min())