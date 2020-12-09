# TODO: place it somewhere better
def reshape_to_4D_if_5D(tensor):
    if len(tensor.shape) == 5:
        return tensor.view(-1, *tensor.shape[2:])
    return tensor