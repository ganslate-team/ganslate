import random
import numpy as np

# TODO: class?
def focal_random_patch(volume, patch_size=(64,64,64), 
                    focus_around_zxy=None, focus_window_to_volume_proportion=None):
    '''
     volume:           whole CT scan (numpy array)
     patch_size:       size of the 3D volume to be extracted from the original volume
     focus_around_zxy: enables taking a patch from B that is in a similar location to the patch from A
    '''
    patch_shape = np.array(patch_size)
    volume_shape = np.array(volume.shape[-3:])
    # a patch can have a starting coordinate anywhere from where it can fit with the defined patch size
    valid_starting_region = volume_shape - patch_shape

    if focus_around_zxy is None:
        # pick a random starting point in valid region of volume
        z = random.randint(0, valid_starting_region[0])
        x = random.randint(0, valid_starting_region[1])
        y = random.randint(0, valid_starting_region[2])

    else: # take a relative neighbor of patch A in patch B
        # 3D window/neighborhood of focus_around_zxy from which will be randomly selected a new start zxy for B
        focus_window = np.multiply(volume_shape, focus_window_to_volume_proportion).astype(np.int64)
        # the starting position from A is given in relative form (A_start_zxy / A_shape)
        zxy = np.array(focus_around_zxy) * volume_shape  # find start position of A translated in B
        z, x, y = focal_random_zxy(zxy, focus_window, valid_starting_region)
        
    # extract the patch from the volume
    patch = volume[z:z+patch_size[0],
                   x:x+patch_size[1],
                   y:y+patch_size[2]]

    # used only for focus_around_zxy
    relative_zxy = (np.array([z,x,y]) / volume_shape).tolist()
    return patch, relative_zxy


def focal_random_zxy(zxy, window, valid_region):
    selected_zxy = []
    # for each axis
    for idx in range(len(zxy)):
        # find the lowest and highest position between which to focus for this axis
        min_position = int(zxy[idx] - window[idx]/2)
        max_position = int(zxy[idx] + window[idx]/2)
        # if one of the boundaries of the focus is outside of the possible area to sample from, cap it
        min_position = max(0, min_position)
        max_position = min(max_position, valid_region[idx])
        # edge cases (no pun intended)
        if min_position > max_position:
            selected_zxy.append(max_position)
        # regular
        else:
            selected_zxy.append(random.randint(min_position, max_position))
    return selected_zxy