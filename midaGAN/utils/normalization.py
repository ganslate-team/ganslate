import torch

def min_max_normalize(image, min_value, max_value):
    image = (image - min_value) / (max_value - min_value)
    return 2*image - 1

def min_max_denormalize(image, min_value, max_value):
    image += 1
    image /= 2
    image *= (max_value - min_value)
    image += min_value
    return image

def z_score_normalize(tensor, scale_to_range=None, mean_std=None, original_scale=None):
    # TODO: Separate it to normal vs slice-based-on-volume
    """Performs z-score normalization on a tensor and scales to a range if specified.
    Mean and standard deviation should be calculated by the function unless there is a good reason 
    for otherwise - for example, if normalizing a slice using precomputed volume's mean and std.
    Original scale as well is only used when scaling slice values using whole volumes min and max.
    TODO: explain better
    """
    EPSILON=1e-100 

    if mean_std:
        mean = mean_std[0]
        std = mean_std[1]
    else:
        mean = tensor.mean()
        std = tensor.std()

    tensor = (tensor - mean) / std

    if original_scale is None:
        original_scale = (tensor.min(), tensor.max())
    else:
        # Volume's min and max values, normalized
        original_scale = (torch.Tensor(original_scale) - mean) / std
        original_scale = tuple(original_scale)
    
    if scale_to_range:
        delta1 = original_scale[1] - original_scale[0]
        delta2 = scale_to_range[1] - scale_to_range[0]
        tensor = (delta2 * (tensor - original_scale[0]) / delta1) + scale_to_range[0]
        
    return tensor#, original_scale