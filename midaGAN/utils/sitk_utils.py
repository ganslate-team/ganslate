import SimpleITK as sitk 
import torch
import numpy as np

def load(file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(file_path))
    sitk_image = reader.Execute()
    return sitk_image

def write(sitk_image, file_path):
    sitk.WriteImage(sitk_image, str(file_path), True) # True is for useCompression flag

def tensor_to_sitk_image(tensor, origin, spacing, direction, dtype='int16'):
    array = tensor.cpu().numpy().astype(str(dtype))
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image.SetOrigin(origin)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetDirection(direction)
    return sitk_image

def get_size_zxy(sitk_image):
    """Get volume size in torch format (e.g. zxy instead of xyz)."""
    size = sitk_image.GetSize()
    n_dims = len(size) 
    if n_dims == 3: 
        return np.array([size[2], size[0], size[1]])

    elif n_dims == 2:
        return np.array([size[0], size[1]])

    else:
        raise NotImplementedError("Not implemented for {} dimensions.".format(n_dims))

def get_npy(sitk_image):
    return sitk.GetArrayFromImage(sitk_image)

def get_tensor(sitk_image):
    return torch.Tensor(get_npy(sitk_image))

def is_volume_smaller_than(sitk_volume, target_shape):
    volume_size = get_size_zxy(sitk_volume)
    if (volume_size < target_shape).any():
        return True
    return False

def is_image_smaller_than(sitk_volume, target_shape):
    image_size = get_size_zxy(sitk_volume[:, :, 0])
    if (image_size < target_shape).any():
        return True
    return False    

def get_npy_dtype(sitk_image):
    return str(sitk.GetArrayFromImage(sitk_image).dtype)