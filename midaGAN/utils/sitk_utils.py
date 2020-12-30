import SimpleITK as sitk
import torch
import numpy as np


def load(file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(file_path))
    sitk_image = reader.Execute()
    return sitk_image


def write(sitk_image, file_path):
    sitk.WriteImage(sitk_image, str(file_path), True)  # True is for useCompression flag


def tensor_to_sitk_image(tensor, origin, spacing, direction, dtype='int16'):
    array = tensor.cpu().numpy().astype(str(dtype))
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image.SetOrigin(origin)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetDirection(direction)
    return sitk_image


def get_size(sitk_image):
    return sitk_image.GetSize()


def get_torch_like_size(sitk_image):
    """Get volume size in torch format (e.g. zxy instead of xyz)."""
    # TODO: does sitk image to tensor produce zxy or zyx?
    size = get_size(sitk_image)
    if len(size) == 3:
        return (size[2], size[0], size[1])
    elif len(size) == 2:
        return size
    else:
        raise NotImplementedError(f"Not implemented for {len(size)} dimensions.")


def get_npy(sitk_image):
    return sitk.GetArrayFromImage(sitk_image)


def get_tensor(sitk_image):
    return torch.Tensor(get_npy(sitk_image))


def is_image_smaller_than(sitk_image, target_size):
    image_size = get_torch_like_size(sitk_image)
    image_size = np.array(image_size)

    # When checking if a volume is big enough in xy only, discard the z dim.
    if len(image_size) == 3 and len(target_size) == 2:
        image_size = image_size[1:]

    if (image_size < target_size).any():
        return True
    return False


def get_npy_dtype(sitk_image):
    return str(sitk.GetArrayFromImage(sitk_image).dtype)


def slice_image(sitk_image, start=(0,0,0), end=(-1,-1,-1)):
    """"Returns the `sitk_image` sliced from the `start` index (x,y,z) to the `end` index.
    """
    size = sitk_image.GetSize()
    assert len(start) == len(end) == len(size)

    # replace -1 dim index placeholders with the size of that dimension
    end = [size[i] if end[i] == -1 else end[i] for i in range(len(end))]

    slice_filter = sitk.SliceImageFilter()
    slice_filter.SetStart(start)
    slice_filter.SetStop(end)
    return slice_filter.Execute(sitk_image)