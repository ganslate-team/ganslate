import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset

from midaGAN import configs
from midaGAN.utils import sitk_utils
from midaGAN.data.utils.normalization import min_max_normalize

from projects.maastro_hx4_pet_translation.datasets.utils.basic import sitk2np, np2tensor, apply_body_mask


EXTENSIONS = ['.nrrd']

# Body mask settings
OUT_OF_BODY_HU = -1024
OUT_OF_BODY_SUV = 0
HU_THRESHOLD = -300  # TODO: Check table HU and update this


@dataclass
class HX4PETTranslationValDatasetConfig(configs.base.BaseDatasetConfig):
    """
    Note: Val dataset is paired, does not supply ldCT and 
    performs full image inference (i.e. no patches)
    """
    name: str = "HX4PETTranslationValDataset" 
    hu_range: Tuple[int, int] = (-1000, 2000)
    fdg_suv_range: Tuple[int, int] = None   # TODO: Set the range
    hx4_suv_range: Tuple[int, int] = None   # TODO: Set the range


class HX4PETTranslationValDataset(Dataset):
    def __init__(self, conf):
        
        # Image file paths
        root_path = conf.train.dataset.root

        self.image_paths = {'FDG-PET': [], 'pCT': [], 'HX4-PET': [], 'body-mask': []}

        for patient in sorted(os.listdir(root_path)):
            patient_image_paths = {}
            patient_image_paths['FDG-PET'] = f"{root_path}/{patient}/fdg_pet.nrrd"
            patient_image_paths['pCT'] = f"{root_path}/{patient}/pct.nrrd"
            patient_image_paths['HX4-PET'] = f"{root_path}/{patient}/hx4_pet_reg.nrrd"
            patient_image_paths['body-mask'] = f"{root_path}/{patient}/pct_body.nrrd"

            for k in self.image_paths.keys():
                self.image_paths[k].append(patient_image_paths[k])

        self.num_datapoints = len(self.image_paths['FDG-PET'])
        
        # Clipping ranges
        self.hu_min, self.hu_max = conf.train.dataset.hu_range
        self.fdg_suv_min, self.fdg_suv_max = conf.train.dataset.fdg_suv_range
        self.hx4_suv_min, self.hx4_suv_max = conf.train.dataset.hx4_suv_range


    def __len__(self):
        return self.num_datapoints


    def __getitem__(self, index):
        
        # ------------
        # Fetch images
        index = index % self.num_datapoints
                
        image_path = {}
        image_path['FDG-PET'] = self.image_paths['FDG-PET'][index]
        image_path['pCT'] = self.image_paths['pCT'][index]
        image_path['HX4-PET'] = self.image_paths['HX4-PET'][index]
        image_path['body-mask'] = self.image_paths['body-mask'][index]

        # Load NRRD as SimpleITK objects (WHD)
        images = {}
        for k in image_path.keys():
            images[k] = sitk_utils.load(image_path[k])
            

        # ---------------
        # Apply body mask
        
        # Convert to numpy (DHW)
        images = sitk2np(images)

        images = apply_body_mask(images)
        
        # Convert to tensors 
        images = np2tensor(images)
        

        # ---------------
        # Clip and normalize

        images['FDG-PET'] = torch.clamp(images['FDG-PET'], self.fdg_suv_min, self.fdg_suv_max)
        images['FDG-PET'] = min_max_normalize(images['FDG-PET'], self.fdg_suv_min, self.fdg_suv_max)

        images['pCT'] = torch.clamp(images['pCT'], self.hu_min, self.hu_max)
        images['pCT'] = min_max_normalize(images['pCT'], self.hu_min, self.hu_max)

        images['HX4-PET'] = torch.clamp(images['HX4-PET'], self.hx4_suv_min, self.hx4_suv_max)
        images['HX4-PET'] = min_max_normalize(images['HX4-PET'], self.hx4_suv_min, self.hx4_suv_max)

        # Return sample dict - A and B to have dims (C,D,H,W)
        A = torch.stack((images['FDG-PET'], images['pCT']), dim=0)
        B = images['HX4-PET'].unsqueeze(dim=0)

        return {'A': A, 'B': B}
