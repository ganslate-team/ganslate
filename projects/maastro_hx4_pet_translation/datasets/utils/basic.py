import numpy as np
import torch

from midaGAN.utils import sitk_utils
from midaGAN.data.utils.body_mask import get_body_mask



def apply_body_mask(image_dict):
    body_mask = image_dict['body-mask']

    # If body mask doesn't exist (N046), then create one from the available CT using morph. ops
    if body_mask is None:
        ct_image_name = [k for k in image_dict.keys() if 'CT' in k][0]
        body_mask = get_body_mask(image_dict[ct_image_name], HU_THRESHOLD)

    for k in image_dict.keys():
        if 'PET' in k:
            image_dict[k] = np.where(body_mask, image_dict[k], OUT_OF_BODY_SUV)
        elif 'CT' in k:
            image_dict[k] = np.where(body_mask, image_dict[k], OUT_OF_BODY_HU)

    return image_dict


def sitk2np(image_dict):
    # WHD to DHW
    for k in image_dict.keys():
        image_dict[k] = sitk_utils.get_npy(image_dict[k])
    return image_dict

def np2tensor(image_dict):
    for k in image_dict.keys():
        image_dict[k] = torch.tensor(image_dict[k])
    return image_dict
