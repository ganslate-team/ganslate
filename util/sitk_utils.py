import SimpleITK as sitk 
from torch import Tensor

def load(file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(file_path)
    sitk_image = reader.Execute()
    return sitk_image

def get_npy(sitk_image):
    return sitk.GetArrayFromImage(sitk_image)

def get_tensor(sitk_image):
    return Tensor(get_npy(sitk_image))