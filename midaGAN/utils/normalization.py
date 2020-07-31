def normalize_from_hu(image, MIN_B=-1024.0, MAX_B=3072.0):
    # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    image = (image - MIN_B) / (MAX_B - MIN_B)
    return 2*image - 1

def denormalize_to_hu(image, MIN_B=-1024.0, MAX_B=3072.0):
    pass

def z_score_normalize(tensor, scale_to_range=None):
    tensor = (tensor - tensor.mean()) / tensor.std()
    original_scale = (tensor.min(), tensor.max())
    if scale_to_range:
        delta1 = original_scale[1] - original_scale[0]
        delta2 = scale_to_range[1] - scale_to_range[0]
        tensor = (delta2 * (tensor - original_scale[0]) / delta1) + scale_to_range[0]
    return tensor, original_scale