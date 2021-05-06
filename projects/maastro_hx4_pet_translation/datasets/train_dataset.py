"""
TODO list:
- Using hx4_pet_reg.nrrd and ldct_reg.nrrd now for unpaired. Use non-reg ones instead? 
- Should hx4_suv_range be same as fdg_suv_range? Because SUV is the common unit of radioactivity in PET
- What's a good way to use data augmentation ?
- Set proper value for `focal_region_proportion`
"""

import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from midaGAN import configs
from midaGAN.utils import sitk_utils


import projects.maastro_hx4_pet_translation.datasets.utils.patch_samplers as patch_samplers
from projects.maastro_hx4_pet_translation.datasets.utils.basic import (sitk2np, 
                                                                       np2tensor, 
                                                                       apply_body_mask,
                                                                       clip_and_min_max_normalize)


EXTENSIONS = ['.nrrd']


@dataclass
class HX4PETTranslationTrainDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "HX4PETTranslationTrainDataset"
    paired: bool = True   # 'True' for Pix2Pix training
    require_ldct: bool = False  # 'True' only for HX4-CycleGAN-balanced
    patch_size: Tuple[int] = (32, 32, 32)
    patch_sampling: str = 'uniform-random' 
    focal_region_proportion: float = 0.3   #  0.2 causes patch sampling error because of too small focal region
    hu_range: Tuple[int, int] = (-1000, 2000)
    fdg_suv_range: Tuple[float, float] = (0.0, 20.0)  
    hx4_suv_range: Tuple[float, float] = (0.0, 4.5)  


class HX4PETTranslationTrainDataset(Dataset):

    def __init__(self, conf):
        
        self.paired = conf.train.dataset.paired
        self.require_ldct = conf.train.dataset.require_ldct

        # Image file paths
        root_path = conf.train.dataset.root
        self.patient_ids = sorted(os.listdir(root_path))

        self.image_paths = {'FDG-PET': [], 'pCT': [], 'HX4-PET': [], 'body-mask': []}
        if self.require_ldct:  # If ldCT is also required during unpaired training
            self.image_paths['ldCT'] = [] 

        for p_id in self.patient_ids:
            patient_image_paths = {}
            patient_image_paths['FDG-PET'] = f"{root_path}/{p_id}/fdg_pet.nrrd"
            patient_image_paths['pCT'] = f"{root_path}/{p_id}/pct.nrrd"
            patient_image_paths['HX4-PET'] = f"{root_path}/{p_id}/hx4_pet_reg.nrrd"
            patient_image_paths['body-mask'] = f"{root_path}/{p_id}/pct_body.nrrd"
            if self.require_ldct: 
                patient_image_paths['ldCT'] = f"{root_path}/{p_id}/ldct_reg.nrrd"

            for k in self.image_paths.keys():
                self.image_paths[k].append(patient_image_paths[k])

        self.num_datapoints_A = len(self.image_paths['FDG-PET'])
        self.num_datapoints_B = len(self.image_paths['HX4-PET'])
        
        # Clipping ranges
        self.hu_min, self.hu_max = conf.train.dataset.hu_range
        self.fdg_suv_min, self.fdg_suv_max = conf.train.dataset.fdg_suv_range
        self.hx4_suv_min, self.hx4_suv_max = conf.train.dataset.hx4_suv_range

        # Patch sampler setup
        patch_size = np.array(conf.train.dataset.patch_size)
        patch_sampling = conf.train.dataset.patch_sampling
        if self.paired:
            self.patch_sampler = patch_samplers.PairedPatchSampler3D(patch_size, patch_sampling)
        else:
            focal_region_proportion = conf.train.dataset.focal_region_proportion
            self.patch_sampler = patch_samplers.UnpairedPatchSampler3D(patch_size, patch_sampling, focal_region_proportion)


    def __len__(self):
        return max(self.num_datapoints_A, self.num_datapoints_B)


    def __getitem__(self, index):
        
        # ------------
        # Fetch images
        index_A = index % self.num_datapoints_A
        index_B = index_A if self.paired else random.randint(0, self.num_datapoints_B - 1)
        
        image_path_A, image_path_B = {}, {}
        image_path_A['FDG-PET'] = self.image_paths['FDG-PET'][index_A]
        image_path_A['pCT'] = self.image_paths['pCT'][index_A]
        image_path_B['HX4-PET'] = self.image_paths['HX4-PET'][index_B]
        if self.require_ldct:
            image_path_B['ldCT'] = self.image_paths['ldCT'][index_B]

        image_path_A['body-mask'] = self.image_paths['body-mask'][index_A]
        image_path_B['body-mask'] = self.image_paths['body-mask'][index_B]

        # Load NRRD as SimpleITK objects (WHD)
        images_A, images_B = {}, {}
        for k in image_path_A.keys():
            images_A[k] = sitk_utils.load(image_path_A[k])
        for k in image_path_B.keys():
            images_B[k] = sitk_utils.load(image_path_B[k])
        

        # ---------
        # Transform
        # TODO: What's a good way to use data aug ?


        # ---------------
        # Apply body mask
        
        # Convert to numpy (DHW)
        images_A = sitk2np(images_A)
        images_B = sitk2np(images_B)

        images_A = apply_body_mask(images_A)
        images_B = apply_body_mask(images_B)
        

        # -----------
        # Sample patches

        # Get patches
        images_A, images_B = self.patch_sampler.get_patch_pair(images_A, images_B)

        # Convert to tensors 
        images_A = np2tensor(images_A)
        images_B = np2tensor(images_B)


        # -----------------------------
        # Clip and normalize intensties

        images_A['FDG-PET'] = clip_and_min_max_normalize(images_A['FDG-PET'], self.fdg_suv_min, self.fdg_suv_max)
        images_A['pCT'] = clip_and_min_max_normalize(images_A['pCT'], self.hu_min, self.hu_max)
        images_B['HX4-PET'] = clip_and_min_max_normalize(images_B['HX4-PET'], self.hx4_suv_min, self.hx4_suv_max)
        if self.require_ldct:
            images_B['ldCT'] = clip_and_min_max_normalize(images_B['ldCT'], self.hu_min, self.hu_max)


        # ---------------------
        # Construct sample dict  
        
        # A and B need to have dims (C,D,H,W)
        A = torch.stack((images_A['FDG-PET'], images_A['pCT']), dim=0)
        
        if self.require_ldct:
            B = torch.stack((images_B['HX4-PET'], images_B['ldCT']), dim=0)
        else:
            B = images_B['HX4-PET'].unsqueeze(dim=0)

        sample_dict = {'A': A, 'B': B}

        return sample_dict

