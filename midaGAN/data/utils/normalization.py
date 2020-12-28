import torch


def min_max_normalize(image, min_value, max_value):
    image = image.float()
    image = (image - min_value) / (max_value - min_value)
    return 2 * image - 1


def min_max_denormalize(image, min_value, max_value):
    image += 1
    image /= 2
    image *= (max_value - min_value)
    image += min_value
    return image


def z_score_normalize(tensor, scale_to_range=None):
    """Performs z-score normalization on a tensor and scales to a range if specified."""
    mean = tensor.mean()
    std = tensor.std()

    tensor = (tensor - mean) / std

    if scale_to_range:
        delta1 = tensor.max() - tensor.min()
        delta2 = scale_to_range[1] - scale_to_range[0]
        tensor = (delta2 * (tensor - tensor.min()) / delta1) + scale_to_range[0]

    return tensor


def z_score_normalize_with_precomputed_stats(tensor,
                                             mean_std,
                                             original_scale=None,
                                             scale_to_range=None):
    """Performs z-score normalization on a tensor using precomputed mean, standard deviation,
    and, optionally, min-max scale. Optionally scales the normalized values to the specified range.
    This function is useful, e.g., when normalizing a slice using its volume's stats.
    """
    mean = mean_std[0]
    std = mean_std[1]

    tensor = (tensor - mean) / std

    if scale_to_range:
        # Volume's min and max values, normalized
        original_scale = (torch.Tensor(original_scale) - mean) / std

        delta1 = original_scale[1] - original_scale[0]
        delta2 = scale_to_range[1] - scale_to_range[0]
        tensor = (delta2 * (tensor - original_scale[0]) / delta1) + scale_to_range[0]

    return tensor


def get_stats_for_z_score_denormalization(tensor):
    # TODO: to be used in inference
    # take into account that the tensor might have been scaled to range
    pass


def z_score_denormalize():
    pass  # TODO
