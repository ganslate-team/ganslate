import random
import numpy as np
import torch

class SliceSampler:
    """ Stochasting Focal Patching technique achieves spatial correspondance of patches extracted from a pair 
    of volumes by:
        (1) Randomly selecting a slice from volume_A (slice_A)
        (2) Calculating the relative start position of the slice_A 
        (3) Translating the slice_A's relative position in volume_B
        (4) Placing a focal region (a proportion of volume shape) around the focal point
        (5) Randomly selecting a start point in the focal region and extracting the slice_B

    The added stochasticity in steps (4) and (5) aims to account for possible differences in positioning 
    of the object between volumes.
    """
    def __init__(self, patch_size, focal_region_proportion):
        self.patch_size = np.array(patch_size)
        try:
            assert len(self.patch_size) == 2
        except AssertionError as error:
            print("Patch size needs to be 2D, use StochasticFocalPatchSampler for 3D!")
            exit()

        self.focal_region_proportion = focal_region_proportion

    def get_slice_pair(self, volume_A, volume_B):
        """Performs stochasting focal patch sampling. Returns A and B patches."""
        slice_A, relative_focal_point = self.slice_and_focal_point_from_A(volume_A)
        slice_B = self.slice_from_B(volume_B, relative_focal_point)
        return slice_A, slice_B

    def slice_and_focal_point_from_A(self, volume):
        """Return random patch from volume A and its relative start position."""
        z, x, y = self.pick_random_start(volume)
        
        x_end, y_end = [sum(pair) for pair in zip((x, y), self.patch_size)] # start + patch size for each coord
        
        volume = self._pad((0, x_end, y_end), volume)

        slice = volume[z, x:x_end, y:y_end]
        relative_focal_point = self.calculate_relative_focal_point(z, x, y, volume)
        return slice, relative_focal_point

    def slice_from_B(self, volume, relative_focal_point):
        """Return random patch from volume B that is in relative neighborhood of patch_A."""
        z, x, y = self.pick_stochastic_focal_start(volume, relative_focal_point)
        x_end, y_end = [sum(pair) for pair in zip((x, y), self.patch_size)] # start + patch size for each coord
        
        volume = self._pad((0, x_end, y_end), volume)

        slice = volume[z, x:x_end, y:y_end]
        return slice

    def pick_random_start(self, volume):
        """Pick a starting point of a patch randomly. Used for patch_A."""
        valid_start_region = self.calculate_valid_start_region(volume)
        z, x, y = [random.randint(0, v) for v in valid_start_region]
        return z, x, y
    
    def pick_stochastic_focal_start(self, volume, relative_focal_point):
        """Pick a starting point of a patch with regards to the focal point neighborhood. Used for patch_B."""
        volume_size = self.get_size(volume)
        focal_region = self.focal_region_proportion * volume_size
        focal_region = focal_region.astype(np.int64)

        focal_point = relative_focal_point * volume_size  # map relative point to corresponding point in this volume
        valid_start_region = self.calculate_valid_start_region(volume)

        z, x, y = self.apply_stochastic_focal_method(focal_point, focal_region, valid_start_region)
        return z, x, y
        
    def apply_stochastic_focal_method(self, focal_point, focal_region, valid_start_region):
        """Applies the focal region window around the focal point and randomly selects the final starting point."""
        start_point = []

        for axis in range(len(focal_point)):
            # find the lowest and highest position between which to focus for this axis
            min_position = int(focal_point[axis] - focal_region[axis]/2)
            max_position = int(focal_point[axis] + focal_region[axis]/2)

            # if one of the boundaries of the focus is outside of the possible area to sample from, cap it
            min_position = max(0, min_position)
            max_position = min(max_position, valid_start_region[axis])
            
            if min_position > max_position: # edge cases # TODO: is it because there's no min(min_position, valid_start_region[axis])
                start_point.append(max_position)
            else:
                start_point.append(random.randint(min_position, max_position))  # regular case

        return start_point

    def calculate_relative_focal_point(self, z, x, y, volume):
        """Relative location of starting point. Obtained by dividing position coordinates with volume size"""
        volume_size = self.get_size(volume)
        focal_point = np.array([z,x,y])

        relative_focal_point = focal_point / volume_size
        return relative_focal_point

    def calculate_valid_start_region(self, volume):
        """Patch can have a starting coordinate anywhere from where it can fit with the defined patch size."""
        volume_size = self.get_size(volume)
        valid_start_region = volume_size - [1, *self.patch_size]
        valid_start_region = np.where(valid_start_region < 0, 0, valid_start_region)

        return valid_start_region


    def _pad(self, index, volume):

        if index[0] > volume.shape[0]:
             volume = np.pad(volume, ((0, index[0] - volume.shape[0]), (0, 0), (0, 0)), 'minimum')

        if index[1] > volume.shape[1]:
            volume = np.pad(volume, ((0, 0), (0, index[1] - volume.shape[1]), (0, 0)), 'minimum')
        
        if index[2] > volume.shape[2]:
            volume = np.pad(volume, ((0, 0), (0, 0), (0, index[2] - volume.shape[2])), 'minimum')

        return volume


    def get_size(self, volume):
        return np.array(volume.shape[-3:])  # last three dimension (Z,X,Y)

