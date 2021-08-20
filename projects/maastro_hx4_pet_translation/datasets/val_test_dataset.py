import os
from dataclasses import dataclass
from typing import Tuple
from loguru import logger

import pandas as pd
import torch
from torch.utils.data import Dataset

from midaGAN import configs
from midaGAN.utils import sitk_utils
from midaGAN.data.utils.normalization import min_max_denormalize
from midaGAN.data.utils.ops import pad

from projects.maastro_hx4_pet_translation.datasets.utils.basic import (sitk2np, 
                                                                       np2tensor, 
                                                                       apply_body_mask,
                                                                       clip_and_min_max_normalize)


@dataclass
class HX4PETTranslationValTestDatasetConfig(configs.base.BaseDatasetConfig):
    """
    Note: Val dataset is paired, and does not supply ldCT
    """
    name: str = "HX4PETTranslationValTestDataset" 
    hu_range: Tuple[int, int] = (-1000, 2000)
    fdg_suv_range: Tuple[float, float] = (0.0, 15.0)  
    hx4_tbr_range: Tuple[float, float] = (0.0, 3.0)
    # Use sliding window inference - If True, the val test engine takes care of it. 
    # Patch size value is interpolated from training patch size
    use_patch_based_inference: bool = False        
    # Option to supply body and GTV masks in the sample_dict. If supplied, masked metrics will
    # be computed additionally during validation which would slow down the training. 
    supply_masks: bool = False
    # Is the model HX4CycleGANBalanced? If so, need to do a small hack while supplying HX4-PET 
    model_is_hx4_cyclegan_balanced: bool = False


class HX4PETTranslationValTestDataset(Dataset):

    def __init__(self, conf):
        
        # Image file paths
        root_path = conf.val.dataset.root

        self.patient_ids = sorted(os.listdir(root_path))
        self.image_paths = {'FDG-PET': [], 'pCT': [], 'HX4-PET': [], 'body-mask': [], 'gtv-mask': []}
        
        for p_id in self.patient_ids:
            patient_image_paths = {}
            patient_image_paths['FDG-PET'] = f"{root_path}/{p_id}/fdg_pet.nrrd"
            patient_image_paths['pCT'] = f"{root_path}/{p_id}/pct.nrrd"
            patient_image_paths['HX4-PET'] = f"{root_path}/{p_id}/hx4_pet_reg.nrrd"
            patient_image_paths['body-mask'] = f"{root_path}/{p_id}/pct_body.nrrd"  
            patient_image_paths['gtv-mask'] = f"{root_path}/{p_id}/pct_gtv.nrrd"

            for k in self.image_paths.keys():
                self.image_paths[k].append(patient_image_paths[k])

        self.num_datapoints = len(self.image_paths['FDG-PET'])
        
        # SUVmean_aorta values for normalizing HX4-PET SUV to TBR
        suv_aorta_mean_file =  f"{os.path.dirname(root_path)}/SUVmean_aorta_HX4.csv"
        self.suv_aorta_mean_values = pd.read_csv(suv_aorta_mean_file, index_col=0)
        self.suv_aorta_mean_values = self.suv_aorta_mean_values.to_dict()['HX4 aorta SUVmean baseline']

        # Clipping ranges
        self.hu_min, self.hu_max = conf.val.dataset.hu_range
        self.fdg_suv_min, self.fdg_suv_max = conf.val.dataset.fdg_suv_range
        self.hx4_tbr_min, self.hx4_tbr_max = conf.val.dataset.hx4_tbr_range

        # Using sliding window inferer or performing full-image inference ?
        self.use_patch_based_inference = conf.val.dataset.use_patch_based_inference        

        # Supply body and GTV masks ?
        self.supply_masks = conf.val.dataset.supply_masks

        # Is HX4-CycleGAN-balanced the model being validated/tested ?
        self.model_is_hx4_cyclegan_balanced = conf.val.dataset.model_is_hx4_cyclegan_balanced        


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
        image_path['gtv-mask'] = self.image_paths['gtv-mask'][index]

        # Load NRRD as SimpleITK objects (WHD)
        images = {}
        for k in image_path.keys():
            # One patient in val set (N046) doesn't have a pCT body mask
            try: 
                images[k] = sitk_utils.load(image_path[k])         
            except RuntimeError:
                if k == 'body-mask':
                    logger.warning(f"Patient {self.patient_ids[index]} does not have a body mask. It will be generated automatically")  
                    # Set as `None` for now, handle it later in apply_body_mask() 
                    # by creating a mask on the go using thresholding
                    images[k] = None
            

        # ----------------------
        # Collect image metadata
        metadata = {
            'patient_id': self.patient_ids[index],
            'size': images['FDG-PET'].GetSize(),
            'origin': images['FDG-PET'].GetOrigin(),
            'spacing': images['FDG-PET'].GetSpacing(),
            'direction': images['FDG-PET'].GetDirection(),
            'dtype': sitk_utils.get_npy_dtype(images['FDG-PET'])
        }


        # ---------------
        # Apply body mask
        
        # Convert to numpy (DHW)
        images = sitk2np(images)

        if self.patient_ids[index] == 'N046':
            generate_body_mask = True
        else:
            generate_body_mask = False

        images = apply_body_mask(images, generate_body_mask)
        
        
        # --------------------
        # Pad images if needed

        # If doing full-image inference, pad images to have a standard size of (64, 512, 512)
        # to avoid issues with UNet's up- and downsampling
        if not self.use_patch_based_inference:
            for k in images.keys():
                images[k] = pad(images[k], target_shape=(64, 512, 512))

        # Convert to tensors 
        images = np2tensor(images)

        # -------------
        # Normalization

        # Normalize HX4-PET SUVs with SUVmean_aorta
        patient_id = self.patient_ids[index]
        images['HX4-PET'] = images['HX4-PET'] / self.suv_aorta_mean_values[patient_id]

        # Clip and then rescale all intensties to range [-1, 1]
        images['FDG-PET'] = clip_and_min_max_normalize(images['FDG-PET'], self.fdg_suv_min, self.fdg_suv_max)
        images['pCT'] = clip_and_min_max_normalize(images['pCT'], self.hu_min, self.hu_max)
        images['HX4-PET'] = clip_and_min_max_normalize(images['HX4-PET'], self.hx4_tbr_min, self.hx4_tbr_max)


        # ---------------------
        # Construct sample dict 

        # A and B need to have dims (C,D,H,W)
        A = torch.stack((images['FDG-PET'], images['pCT']), dim=0)

        if self.model_is_hx4_cyclegan_balanced:
            # Create a dummy array to fill up the 2nd channel
            zeros_dummy = torch.zeros_like(images['HX4-PET'])
            B =  torch.stack([images['HX4-PET'], zeros_dummy], dim=0)
        else:
            B = images['HX4-PET'].unsqueeze(dim=0)
        
        sample_dict = {'A': A, 'B': B}
        
        # Include masks, if needed
        if self.supply_masks:
            sample_dict['masks'] = {'BODY': images['body-mask'].unsqueeze(dim=0), 
                                    'GTV': images['gtv-mask'].unsqueeze(dim=0)}
        
        # Include metadata
        sample_dict['metadata'] = metadata

        return sample_dict


    def denormalize(self, tensor):
        """Allows the Tester and Validator to calculate the metrics in
        the original range of values.
        `tensor` can be either the predicted or the ground truth HX4-PET image tensor
        """
        tensor = min_max_denormalize(tensor, self.hx4_tbr_min, self.hx4_tbr_max)
        return tensor


    def save(self, tensor, save_dir, metadata):
        """ Save predicted tensors as NRRD
        """
        
        # If the model is HX4-CycleGAN-balanced, tensor is 2-channel with the 
        # 1st channel containing HX4-PET and 2nd channel containing a dummy array. 
        if self.model_is_hx4_cyclegan_balanced:
            tensor = tensor[0]  # Dim 1 is the channel dim
        else:
            tensor = tensor.squeeze()

        # Rescale back to [self.hx4_tbr_min, self.hx4_tbr_max]
        tensor = min_max_denormalize(tensor.cpu(), self.hx4_tbr_min, self.hx4_tbr_max)
        
        # Denormalize TBR to SUV
        patient_id = metadata['patient_id']
        tensor = tensor * self.suv_aorta_mean_values[patient_id]

        sitk_image = sitk_utils.tensor_to_sitk_image(tensor, metadata['origin'],
                                                     metadata['spacing'], metadata['direction'],
                                                     metadata['dtype'])
        # Write to file
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{patient_id}.nrrd"
        sitk_utils.write(sitk_image, save_path)
