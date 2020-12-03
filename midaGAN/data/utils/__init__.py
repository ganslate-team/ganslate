# coding=utf-8
# Copyright (c) midaGAN Contributors
import random 
import numpy as np

from midaGAN.utils import sitk_utils
import logging

logger = logging.getLogger(__name__)

def size_invalid_check_and_replace(volume, patch_size, replacement_paths=[], original_path=None):
    """
    Check if volume loaded is invalid, if so replace with another volume from the same patient. 

    Parameters
    ----------------
    volume: Input volume to check for validity
    patch_size: Patch size to compare against. If volume smaller than patch size, it is considered invalid
    replacement_paths: List of paths to sample from if volume is invalid
    original_path: Path of current volume. Used to remove entry


    Returns
    ----------------
    volume or None

    """

    if original_path is not None:
        replacement_paths.remove(original_path)

    # Check if volume is smaller than patch size
    
    if len(patch_size) == 3:
        fn = eval(f"sitk_utils.is_volume_smaller_than")
    elif len(patch_size) == 2:
        fn = eval(f"sitk_utils.is_image_smaller_than")
    else:
        raise NotImplementedError()

    while fn(volume, patch_size):
        logger.warning(f"Volume size smaller than the defined patch size.\
            Volume: {sitk_utils.get_size_zxy(volume)} \npatch_size: {patch_size}. \n \
            Volume path: {original_path}")

        logger.warning(f"Replacing with random choice from: {replacement_paths}")
        
        if len(replacement_paths) == 0:
            return None

        # Load volume randomly from replacement paths
        path = random.choice(replacement_paths)
        logger.warning(f"Loading replacement scan from {path}")
        volume = sitk_utils.load(path)  

        # Remove current path from replacement paths
        replacement_paths.remove(path)

    return volume


def pad(index, volume):
    pad_value = volume.min()

    if index[0] > volume.shape[0]:
        
        pad = (index[0] - volume.shape[0])
        if pad % 2 == 0:
            pad_before = pad // 2
            pad_after = pad // 2

        else:
            pad_before = pad // 2
            pad_after = (pad // 2) + 1

        volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), 'constant', constant_values=pad_value)

    if index[1] > volume.shape[1]:

        pad = (index[1] - volume.shape[1])
        if pad % 2 == 0:
            pad_before = pad // 2
            pad_after = pad // 2

        else:
            pad_before = pad // 2
            pad_after = (pad // 2) + 1 

        volume = np.pad(volume, ((0, 0), (pad_before, pad_after), (0, 0)), 'constant', constant_values=pad_value)
    
    if index[2] > volume.shape[2]:
        
        pad = (index[2] - volume.shape[2])
        
        if pad % 2 == 0:
            pad_before = pad // 2
            pad_after = pad // 2

        else:
            pad_before = pad // 2
            pad_after = (pad // 2) + 1    

        volume = np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)), 'constant', constant_values=pad_value)

    return volume