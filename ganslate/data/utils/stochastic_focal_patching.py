import random
import numpy as np


class StochasticFocalPatchSampler:
    """ Stochasting Focal Patching technique achieves spatial correspondance of patches extracted from a pair
    of volumes by:
        (1) Randomly selecting a patch from volume_A (patch_A)
        (2) Calculating the relative start position of the patch_A
        (3) Translating the patch_A's relative position in volume_B
        (4) Placing a focal region (a proportion of volume shape) around the focal point
        (5) Randomly selecting a start point in the focal region and extracting the patch_B

    The added stochasticity in steps (4) and (5) aims to account for possible differences in positioning
    of the object between volumes.
    """

    def __init__(self, patch_size, focal_region_proportion):
        self.focal_region_proportion = focal_region_proportion
        self.dims = len(patch_size)
        # If patch size is 2D, expand it to [1, *patch_size]
        if self.dims == 2:
            patch_size = [1, *patch_size]

        self.patch_size = np.array(patch_size)

    def get_patch_pair(self, volume_A, volume_B):
        """Performs stochasting focal patch sampling. Returns A and B patches."""
        patch_A, relative_focal_point = self.patch_and_focal_point_from_A(volume_A)
        patch_B = self.patch_from_B(volume_B, relative_focal_point)

        # If input patch size is 2D, compress the depth dim to return [H, W]
        if self.dims == 2:
            patch_A, patch_B = patch_A.squeeze(0), patch_B.squeeze(0)

        return patch_A, patch_B

    def patch_and_focal_point_from_A(self, volume):
        """Return random patch from volume A and its relative start position."""
        z, x, y = self.pick_random_start(volume)
        # start + patch size for each coord
        z_end, x_end, y_end = [sum(pair) for pair in zip((z, x, y), self.patch_size)]

        patch = volume[z:z_end, x:x_end, y:y_end]
        relative_focal_point = self.calculate_relative_focal_point(z, x, y, volume)
        return patch, relative_focal_point

    def patch_from_B(self, volume, relative_focal_point):
        """Return random patch from volume B that is in relative neighborhood of patch_A."""
        z, x, y = self.pick_stochastic_focal_start(volume, relative_focal_point)
        # start + patch size for each coord
        z_end, x_end, y_end = [sum(pair) for pair in zip((z, x, y), self.patch_size)]

        patch = volume[z:z_end, x:x_end, y:y_end]
        return patch

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

        # Map relative point to corresponding point in this volume
        focal_point = relative_focal_point * volume_size
        valid_start_region = self.calculate_valid_start_region(volume)

        z, x, y = self.apply_stochastic_focal_method(focal_point, focal_region, valid_start_region)
        return z, x, y

    def apply_stochastic_focal_method(self, focal_point, focal_region, valid_start_region):
        """Applies the focal region window around the focal point and randomly selects the final starting point."""
        start_point = []

        for axis in range(len(focal_point)):
            # find the lowest and highest position between which to focus for this axis
            min_position = int(focal_point[axis] - focal_region[axis] / 2)
            max_position = int(focal_point[axis] + focal_region[axis] / 2)

            # If one of the boundaries of the focus is outside of the possible area to sample from, cap it
            min_position = max(0, min_position)
            max_position = min(max_position, valid_start_region[axis])

            # Edge cases # TODO: is it because there's no min(min_position, valid_start_region[axis])
            if min_position > max_position:
                start_point.append(max_position)
            else:
                start_point.append(random.randint(min_position, max_position))

        return start_point

    def calculate_relative_focal_point(self, z, x, y, volume):
        """Relative location of starting point. Obtained by dividing position coordinates with volume size"""
        volume_size = self.get_size(volume)
        focal_point = np.array([z, x, y])

        relative_focal_point = focal_point / volume_size
        return relative_focal_point

    def calculate_valid_start_region(self, volume):
        """Patch can have a starting coordinate anywhere from where it can fit with the defined patch size."""
        volume_size = self.get_size(volume)
        valid_start_region = volume_size - self.patch_size

        if np.any(valid_start_region < 0):
            raise RuntimeError(
                f"The volume, {volume_size} provided to the sampler is smaller than the patch size: {self.patch_size}"
            )

        return valid_start_region

    def get_size(self, volume):
        # last three dimension (Z,X,Y)
        return np.array(volume.shape[-3:])
