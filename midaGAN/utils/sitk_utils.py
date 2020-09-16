import SimpleITK as sitk 
from torch import Tensor
import numpy as np

def load(file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(file_path))
    sitk_image = reader.Execute()
    return sitk_image

def get_size_zxy(sitk_image):
    """Get volume size in torch format (e.g. zxy instead of xyz)."""
    size = sitk_image.GetSize()
    n_dims = len(size) 
    if n_dims == 3: 
        return np.array([size[2], size[0], size[1]])
    else:
        raise NotImplementedError("Not implemented for {} dimensions.".format(n_dims))

def get_npy(sitk_image):
    return sitk.GetArrayFromImage(sitk_image)

def get_tensor(sitk_image):
    return Tensor(get_npy(sitk_image))

def is_volume_smaller_than(self, sitk_volume, target_shape):
    volume_size = get_size_zxy(sitk_volume)
    if (volume_size < target_shape).any():
        return True
    return False