import torch
import numpy as np


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


def unequal_normalize(x, min_value, max_value, split_points, split_proportions):

    '''
    Normalizes given data according to a three-piecewise linear function.
    '''
    max_range = max_value - min_value

    def inner(x):
        # get range up to split point
        limited_range = split_points[0] - min_value
        range_proportion = limited_range/max_range
        # To normalize input to range of [0, 1]
        x = (x - min_value)/max_range
        # To normalize according to desired split percentage
        range_modifier = split_proportions[0]/range_proportion
        x *= range_modifier
        return 2*x-1

    def middle(x):
        # get range between split points
        limited_range = split_points[1] - split_points[0]
        range_proportion = limited_range/max_range
        # To normalize input to range of [0, 1]
        x = (x - min_value)/max_range
        # To normalize according to desired split percentage
        range_modifier = split_proportions[1]/range_proportion
        x = x*range_modifier
        # have to add previous piecewise function max and deduct current piecewise function min
        current_min = (split_points[0]- min_value)/max_range*range_modifier

        range_modifier_inner = split_proportions[0]/((split_points[0] - min_value)/max_range)
        previous_max = (split_points[0]- min_value)/max_range*range_modifier_inner

        x = x + previous_max - current_min
        return 2*x-1
    
    def outer(x):
        # get range between split points
        limited_range = max_value - split_points[1]
        range_proportion = limited_range/max_range
        # To normalize input to range of [0, 1]
        x = (x - min_value)/max_range
        # To normalize according to desired split percentage
        range_modifier = split_proportions[2]/range_proportion
        x = x*range_modifier
        # have to add previous piecewise functions max and deduct previous and current piecewise function min
        current_min = (split_points[1]- min_value)/max_range*range_modifier

        range_modifier_inner = split_proportions[0]/((split_points[0] - min_value)/max_range)
        previous_max_inner = (split_points[0]- min_value)/max_range*range_modifier_inner
        
        range_modifier_middle = split_proportions[1]/((split_points[1] - split_points[0])/max_range)
        previous_max_middle = (split_points[1]- min_value)/max_range*range_modifier_middle
        previous_min_inner = (split_points[0]- min_value)/max_range*range_modifier_middle

        x = x + previous_max_inner + previous_max_middle - current_min - previous_min_inner
        return 2*x-1

    result = np.piecewise(x, [x < split_points[0], (x >= split_points[0]) & (x <= split_points[1]), x > split_points[1]], [lambda x: inner(x), lambda x: middle(x), lambda x: outer(x)])
    return result


def unequal_denormalize(x, min_value, max_value, split_points, split_proportions):

    '''
    Denormalizes given data according to a three-piecewise linear function.
    '''

    split_point_1 = (2*split_proportions[0]-1)
    split_point_2 = (2*(split_proportions[1]+split_proportions[0])-1)
    max_range = max_value - min_value

    def inner(x):
        # get data to [0,1]
        x = (x+1)/2
        # divide by range modifier
        limited_range = split_points[0] - min_value
        range_proportion = limited_range/max_range
        range_modifier = split_proportions[0]/range_proportion
        x /= range_modifier
        # denorm from [0,1] normalizer
        x *= max_range
        x += min_value
        return x

    def middle(x):
        # get data to [0,1]
        x = (x+1)/2
        # deduct previous piecewise function max and add current piecewise function min
        limited_range = split_points[1] - split_points[0]
        range_proportion = limited_range/max_range
        range_modifier = split_proportions[1]/range_proportion

        current_min = (split_points[0]- min_value)/max_range*range_modifier

        range_modifier_inner = split_proportions[0]/((split_points[0] - min_value)/max_range)
        previous_max = (split_points[0]- min_value)/max_range*range_modifier_inner

        x = x - previous_max + current_min

        # Denormalize according to desired split percentage
        x = x/range_modifier

        # denorm from [0,1] normalizer
        x *= max_range
        x += min_value
        return x
    
    def outer(x):
        # get data to [0,1]
        x = (x+1)/2
        # get range between split points and range_modifier
        limited_range = max_value - split_points[1]
        range_proportion = limited_range/max_range
        range_modifier = split_proportions[2]/range_proportion

        # Deduct previous piecewise functions max and add previous and current piecewise function min
        current_min = (split_points[1]- min_value)/max_range*range_modifier

        range_modifier_inner = split_proportions[0]/((split_points[0] - min_value)/max_range)
        previous_max_inner = (split_points[0]- min_value)/max_range*range_modifier_inner
        
        range_modifier_middle = split_proportions[1]/((split_points[1] - split_points[0])/max_range)
        previous_max_middle = (split_points[1]- min_value)/max_range*range_modifier_middle
        previous_min_inner = (split_points[0]- min_value)/max_range*range_modifier_middle

        x = x - previous_max_inner - previous_max_middle + current_min + previous_min_inner
        # denormalize according to split percentage
        x = x/range_modifier
        # denorm from [0,1] normalizer
        x *= max_range
        x += min_value
        return x

    result = np.piecewise(x, [x < split_point_1, (x >= split_point_1) & (x <= split_point_2), x > split_point_2], [lambda x: inner(x), lambda x: middle(x), lambda x: outer(x)])
    return result