# coding=utf-8
# Copyright (c) midaGAN Contributors

from midaGAN.utils import sitk_utils
import logging

logger = logging.getLogger(__name__)

def volume_invalid_check_and_replace(volume, patch_size, replacement_paths=[], original_path=None):
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
    while sitk_utils.is_volume_smaller_than(volume, patch_size):
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
