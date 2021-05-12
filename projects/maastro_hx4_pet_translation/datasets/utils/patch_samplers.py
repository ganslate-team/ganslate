import numpy as np


PAIRED_SAMPLING_SCHEMES = ('uniform-random-within-body', 'fdg-pet-weighted')
UNPAIRED_SAMPLING_SCHEMES = ('uniform-random-within-body-sf', 'fdg-pet-weighted-sf')


class PairedPatchSampler3D():
    """3D patch sampler for paired training.

    Available patch sampling schemes:
        1. 'uniform-random-within-body'
        2. 'fdg-pet-weighted'
    """
    def __init__(self, patch_size, sampling):

        if sampling not in PAIRED_SAMPLING_SCHEMES:
            raise ValueError(f"`{sampling}` not a valid paired patch sampling scheme. \
                               Available schemes: {PAIRED_SAMPLING_SCHEMES}")

        self.patch_size = np.array(patch_size)
        self.sampling = sampling


    def get_patch_pair(self, image_dict_A, image_dict_B):
        
        # Sample a single focal point to be used for both domain A and B images
        # Domain A and domain B images are expected to be voxel-to-voxel paired
        focal_point = self._sample_common_focal_point(image_dict_A)

        # Extract patches from all volumes given this focal point and the patch size
        start_idx = focal_point - np.floor(self.patch_size/2)
        end_idx = start_idx + self.patch_size
        z1, y1, x1 = start_idx.astype(np.uint16)
        z2, y2, x2 = end_idx.astype(np.uint16)
    
        patch_dict_A, patch_dict_B = {}, {}
        for k in image_dict_A.keys():
            patch_dict_A[k] = image_dict_A[k][z1:z2, y1:y2, x1:x2]
        for k in image_dict_B.keys():
            patch_dict_B[k] = image_dict_B[k][z1:z2, y1:y2, x1:x2]
        
        return patch_dict_A, patch_dict_B


    def _sample_common_focal_point(self, image_dict_A):        
        body_mask = image_dict_A['body-mask']
        volume_size = body_mask.shape[-3:]  # DHW

        # Initialize sampling probability map as a volumetric mask of body region contained inside the 
        # volume's valid patch region (i.e. suffieciently away from the volume borders)
        sampling_prob_map = init_sampling_probability_map(volume_size, self.patch_size, body_mask)

        # Depending on the sampling technique, construct the probability map
        if self.sampling == 'uniform-random-within-body':
            # Uniform random over all valid focal points
            sampling_prob_map = sampling_prob_map / np.sum(sampling_prob_map)

        elif self.sampling == 'fdg-pet-weighted':
            # Random sampling, biased to high SUV regions in FDG-PET
            FDG_PET_volume = image_dict_A['FDG-PET']
            # Clip negative values to zero 
            FDG_PET_volume = np.clip(FDG_PET_volume, 0, None)
            # Update the probability map
            sampling_prob_map = sampling_prob_map * FDG_PET_volume
            sampling_prob_map = sampling_prob_map / np.sum(sampling_prob_map)

        # Sample focal points using this probability map
        focal_point = sample_from_probability_map(sampling_prob_map)

        return np.array(focal_point).astype(np.uint16)



class UnpairedPatchSampler3D():
    """3D patch sampler for unpaired training.

    Variations of Stochastic Focal patch sampling, where different schemes 
    differ in the way the focal point is sampled from domain A image(s).
    Essentially, the schemes provide different prior probability distributions
    to sample from.

    Available patch sampling schemes:
        1. 'uniform-random-sf'
        2. 'fdg-pet-weighted-sf'
    """
    def __init__(self, patch_size, sampling, focal_region_proportion):

        if sampling not in UNPAIRED_SAMPLING_SCHEMES:
            raise ValueError(f"`{sampling}` not a valid unpaired patch sampling scheme. \
                               Available schemes: {UNPAIRED_SAMPLING_SCHEMES}")

        self.patch_size = np.array(patch_size)
        self.sampling = sampling
        self.focal_region_proportion = focal_region_proportion


    def get_patch_pair(self, image_dict_A, image_dict_B):
        # Sample a focal point and its size-normlaized version for domain A images
        focal_point_A, relative_focal_point = self._sample_focal_point_A(image_dict_A)

        # Sample a focal point for B images that is in relative neighborhood of the focal point of A images
        focal_point_B = self._sample_focal_point_B(image_dict_B, relative_focal_point)

        # Extract patches from all volumes given this focal point and the patch size
        start_idx_A = focal_point_A - np.floor(self.patch_size/2)
        end_idx_A = start_idx_A + self.patch_size
        z1_A, y1_A, x1_A = start_idx_A.astype(np.uint16)
        z2_A, y2_A, x2_A = end_idx_A.astype(np.uint16)

        start_idx_B = focal_point_B - np.floor(self.patch_size/2)
        end_idx_B = start_idx_B + self.patch_size
        z1_B, y1_B, x1_B = start_idx_B.astype(np.uint16)
        z2_B, y2_B, x2_B = end_idx_B.astype(np.uint16)
    
        patch_dict_A = {}
        for k in image_dict_A.keys():
            patch_dict_A[k] = image_dict_A[k][z1_A:z2_A, y1_A:y2_A, x1_A:x2_A]
        
        patch_dict_B = {}
        for k in image_dict_B.keys():
            patch_dict_B[k] = image_dict_B[k][z1_B:z2_B, y1_B:y2_B, x1_B:x2_B]
        
        return patch_dict_A, patch_dict_B


    def _sample_focal_point_A(self, image_dict_A):
        body_mask = image_dict_A['body-mask']
        volume_size = body_mask.shape  # DHW

        # Initialize sampling probability map as a volumetric mask of body region contained inside the 
        # volume's valid patch region (i.e. suffieciently away from the volume borders)
        sampling_prob_map = sampling_prob_map = init_sampling_probability_map(volume_size, self.patch_size, body_mask)

        # Depending on the sampling technique, construct the probability map
        if self.sampling == 'uniform-random-within-body-sf':
            # Uniform random over all valid focal points
            sampling_prob_map = sampling_prob_map / np.sum(sampling_prob_map)

        elif self.sampling == 'fdg-pet-weighted-sf':
            # Random sampling, biased to high SUV regions in FDG-PET
            FDG_PET_volume = image_dict_A['FDG-PET']
            # Clip negative values to zero 
            FDG_PET_volume = np.clip(FDG_PET_volume, 0, None)
            # Update the probability map
            sampling_prob_map = sampling_prob_map * FDG_PET_volume
            sampling_prob_map = sampling_prob_map / np.sum(sampling_prob_map)

        # Sample focal point using this probability map
        focal_point = sample_from_probability_map(sampling_prob_map)
        focal_point = np.array(focal_point)

        # Calculate the relative focal point by normalizing focal point indices with the volume size
        relative_focal_point = focal_point / np.array(volume_size)

        return focal_point.astype(np.uint16), relative_focal_point


    def _sample_focal_point_B(self, image_dict_B, relative_focal_point):
        body_mask = image_dict_B['body-mask']
        volume_size = body_mask.shape  # DHW

        focal_region_size = self.focal_region_proportion * np.array(volume_size)
        focal_region_size = focal_region_size.astype(np.uint16)

        # Map relative point to corresponding point in this volume
        focal_point = relative_focal_point * np.array(volume_size)

        # Intialize a sampling probability map for domain B images
        sampling_prob_map = init_sampling_probability_map(volume_size, self.patch_size, body_mask)

        # Apply Stochastic focal sampling
        focal_point_after_sf = self._apply_stochastic_focal_method(focal_point, focal_region_size, sampling_prob_map)
        return focal_point_after_sf


    def _apply_stochastic_focal_method(self, focal_point, focal_region_size, sampling_prob_map):
        
        # Create a focal region mask having the same size as the volume      
        volume_size = sampling_prob_map.shape
        valid_region_min, valid_region_max = get_valid_region_corner_points(volume_size, self.patch_size)
        
        focal_region_min, focal_region_max = [], [] 
        
        for axis in range(len(focal_point)):
            # Find the lowest and highest position between which to focus for this axis
            min_position = int(focal_point[axis] - focal_region_size[axis] / 2)
            max_position = int(focal_point[axis] + focal_region_size[axis] / 2)            

            # If one of the boundaries of the focus is outside of the valid area, cap it
            min_position = max(min_position, valid_region_min[axis])
            max_position = min(max_position, valid_region_max[axis])

            focal_region_min.append(min_position)
            focal_region_max.append(max_position)

        z_min, y_min, x_min = focal_region_min
        z_max, y_max, x_max = focal_region_max        

        # Check whether or not the focal region limits are reasonable
        if z_min >= z_max or y_min >= y_max or x_min >= x_max:
            raise RuntimeError("Focal region couldn't be properly defined. \
                Likely causes: Specified `focal_region_proportion` too small.")

        focal_region_mask = np.zeros_like(sampling_prob_map)
        focal_region_mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1

        # Update the sampling map by taking the intersection with the focal region mask.
        # This is to make sure the sampled focal point is:  
        #   1. Within the volume's valid region  
        #   2. AND, Within body region 
        #   3. AND, Within focal region
        sampling_prob_map = sampling_prob_map * focal_region_mask
        sampling_prob_map = sampling_prob_map / np.sum(sampling_prob_map)

        # Sample focal point using this probability map
        focal_point_after_sf = sample_from_probability_map(sampling_prob_map)
        return focal_point_after_sf



# --------------
# Util functions

def sample_from_probability_map(sampling_prob_map):
    """TODO: Doc
    """
    # Select relevant indices to sample from (i.e. those having a non-zero probability)
    relevant_idxs = np.argwhere(sampling_prob_map > 0)

    # Using the sampling probability map, define the sampling distribution over these relevant indices
    distribution = sampling_prob_map[sampling_prob_map > 0].flatten()
    
    # Sample a single voxel index. This is the focal point.
    s = np.random.choice(len(relevant_idxs), p=distribution)
    sampled_idx = relevant_idxs[s]
    
    return sampled_idx


def init_sampling_probability_map(volume_size, patch_size, body_mask):
    """Initialize sampling probability map as a volumetric mask of body region contained inside the 
    volume's valid patch region (i.e. suffieciently away from the volume borders)
    """
    # Initialize sampling probability map as zeros
    sampling_prob_map = np.zeros(volume_size)

    # Get valid index range for focal points - upper-bound inclusive   
    valid_foc_pt_idx_min, valid_foc_pt_idx_max = get_valid_region_corner_points(volume_size, patch_size)
    z_min, y_min, x_min = valid_foc_pt_idx_min.astype(np.uint16)
    z_max, y_max, x_max = valid_foc_pt_idx_max.astype(np.uint16)

    # Set valid zone values as 1
    sampling_prob_map[z_min:z_max, y_min:y_max, x_min:x_max] = 1

    # Filter out those outside the body region. To avoid sampling patches from the background areas.
    sampling_prob_map = sampling_prob_map * body_mask

    return sampling_prob_map


def get_valid_region_corner_points(volume_size, patch_size):
    valid_foc_pt_idx_min = np.zeros(3) + np.floor(patch_size/2)
    valid_foc_pt_idx_max = np.array(volume_size) - np.ceil(patch_size/2)   
    return valid_foc_pt_idx_min.astype(np.int16), valid_foc_pt_idx_max.astype(np.int16)
