import numpy as np


PAIRED_SAMPLING_SCHEMES = ('uniform-random', 'fdg-pet-weighted')
UNPAIRED_SAMPLING_SCHEMES = ('stochastic-focal')


class PairedPatchSampler3D():
    """ 3D patch sampler for paired training.
    Available patch sampling schemes:
        1. 'uniform-random'
        2. 'fdg-pet-weighted'
    """
    def __init__(self, patch_size, sampling):

        if sampling not in PAIRED_SAMPLING_SCHEMES:
            raise ValueError(f" `{sampling}` not a valid paired patch sampling technique")

        self.patch_size = np.array(patch_size)
        self.sampling = sampling


    def get_patch_pair(self, image_dict_A, image_dict_B):
        
        # Sample a focal point
        focal_point = self._sample_focal_point(image_dict_A, image_dict_B)

        #  Get patches from all volumes given this focal point and the patch size
        start_idx = focal_point - np.floor(self.patch_size/2)
        end_idx = start_idx + self.patch_size
        z1, y1, x1 = start_idx.astype(np.uint16)
        z2, y2, x2 = end_idx.astype(np.uint16)
    
        patch_dict_A, patch_dict_B = {}, {}
        for k in image_dict_A.keys():
            patch_dict_A[k] = image_dict_A[k][z1:z2, y1:y2, x1:x2]
        for k in image_dict_B.keys():
            patch_dict_B[k] = image_dict_B[k][z1:z2, y1:y2, x1:x2]
        
        # Check
        if tuple(patch_dict_A['FDG-PET'].shape[-3:]) != tuple(self.patch_size):
            raise RuntimeError(f"Weird patch size - {patch_dict_A['FDG-PET'].shape} |  Focal point - {focal_point}")

        return patch_dict_A, patch_dict_B


    def _sample_focal_point(self, image_dict_A, image_dict_B):        
        body_mask = image_dict_A['body-mask']
        volume_size = body_mask.shape[-3:]  # DHW

        # Initialize sampling probability map as zeros
        sampling_prob_map = np.zeros_like(body_mask)

        # Get valid index range for focal points - upper-bound inclusive
        valid_foc_pt_idx_min = np.zeros(3) + np.floor(self.patch_size/2)
        valid_foc_pt_idx_max = np.array(volume_size) - np.ceil(self.patch_size/2)   

        z1, y1, x1 = valid_foc_pt_idx_min.astype(np.uint16)
        z2, y2, x2 = valid_foc_pt_idx_max.astype(np.uint16)

        # Set valid zone values as 1
        sampling_prob_map[z1:z2, y1:y2, x1:x2] = 1

        # Filter out those outside the body region
        sampling_prob_map = sampling_prob_map * body_mask

        # Depending on the sampling technique, construct the probability map
        if self.sampling == 'uniform-random':
            # Uniform random over all valid focal points
            sampling_prob_map = sampling_prob_map / np.sum(sampling_prob_map)

        elif self.sampling == 'fdg-pet-weighted':
            # Random sampling, biased to high SUV regions in FDG-PET
            FDG_PET_volume = image_dict_A['FDG-PET']
            # Zero down small values
            suv_threshold = 0.1
            FDG_PET_volume[FDG_PET_volume < suv_threshold] = 0 
            # Probability map
            sampling_prob_map = sampling_prob_map * FDG_PET_volume
            sampling_prob_map = sampling_prob_map / np.sum(sampling_prob_map)

        # Sample focal points using this probability map
        focal_point = sample_from_probability_map(sampling_prob_map)

        return np.array(focal_point).astype(np.uint16)



class UnpairedPatchSampler3D():
    """
    TODO: implement
    """
    def __init__(self):
        pass



def sample_from_probability_map(sampling_prob_map):
    relevant_idxs = np.argwhere(sampling_prob_map > 0)
    distribution = sampling_prob_map[sampling_prob_map > 0].flatten()
    s = np.random.choice(len(relevant_idxs), size=1, p=distribution)[0]
    sampled_idx = relevant_idxs[s]
    return sampled_idx