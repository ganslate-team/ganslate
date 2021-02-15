from pathlib import Path
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

import midaGAN
from midaGAN.utils import io, sitk_utils
from midaGAN.data.utils.normalization import min_max_normalize, min_max_denormalize
from midaGAN.data.utils.registration_methods import register_CT_to_CBCT
from midaGAN.data.utils.body_mask import apply_body_mask

# Config imports
from typing import Tuple
from dataclasses import dataclass
from omegaconf import MISSING
from midaGAN import configs

logger = logging.getLogger(__name__)


@dataclass
class CBCTtoCTValidationDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "CBCTtoCTValidationDataset"
    hounsfield_units_range: Tuple[int, int] = (-1000, 2000)


class CBCTtoCTValidationDataset(Dataset):

    def __init__(self, conf):
        self.root_path = Path(conf.val.dataset.root).resolve()

        self.pairs = []
        for CT_CBCT_pair_dir in self.root_path.iterdir():
            CT_CBCT_pair_dir = self.root_path / CT_CBCT_pair_dir

            CBCT = list(CT_CBCT_pair_dir.rglob('CBCT.nrrd'))[0]
            CT = list(CT_CBCT_pair_dir.rglob('CT.nrrd'))[0]
            self.pairs.append((CBCT, CT))

        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.val.dataset.hounsfield_units_range

    def __getitem__(self, index):
        path_CBCT, path_CT = self.pairs[index]

        CBCT = sitk_utils.load(path_CBCT)
        CT = sitk_utils.load(path_CT)

        metadata = {
            'path': str(path_CBCT),
            'size': CBCT.GetSize(),
            'origin': CBCT.GetOrigin(),
            'spacing': CBCT.GetSpacing(),
            'direction': CBCT.GetDirection(),
            'dtype': sitk_utils.get_npy_dtype(CBCT)
        }

        CT = mask_out_ct(CT, path_CT.parent, self.hu_min)

        ################ TO RECONSIDER LATER ################
        # Limit CT so that it only contains part of the body shown in CBCT
        CT = register_CT_to_CBCT(CT, CBCT)
        CBCT = apply_body_mask(sitk_utils.get_npy(CBCT),
                               apply_mask=True,
                               masking_value=self.hu_min,
                               hu_threshold=-800)

        CT = apply_body_mask(sitk_utils.get_npy(CT),
                             apply_mask=True,
                             masking_value=self.hu_min,
                             hu_threshold=-600)
        #####################################################

        CBCT = torch.tensor(CBCT)
        CT = torch.tensor(CT)

        CBCT, CT = clamp_normalize(CBCT, CT, self.hu_min, self.hu_max)

        return {'A': CBCT, 'B': CT, "metadata": metadata}

    def __len__(self):
        return len(self.pairs)

    def denormalize(self, tensor):
        """Allows the Tester and Validator to calculate the metrics in
        the original range of values.
        """
        # tensor = min_max_denormalize(tensor.clone(), self.hu_min, self.hu_max)

        # # Offset tensor with hu_min to get CT number from HU
        # # Useful for things like PSNR calculations
        # return tensor - self.hu_min


        # TODO: TEMPORARYYYYY
        return (tensor + 1) / 2


    def save(self, tensor, save_dir, metadata):
        tensor = tensor.squeeze().cpu()
        tensor = min_max_denormalize(tensor, self.hu_min, self.hu_max)

        sitk_image = sitk_utils.tensor_to_sitk_image(tensor, metadata['origin'],
                                                     metadata['spacing'], metadata['direction'],
                                                     metadata['dtype'])

        # Dataset used has a directory per each datapoint, the name of each
        # datapoint's dir is used to save the output
        datapoint_path = Path(str(metadata['path']))

        save_path = datapoint_path.relative_to(self.root_path)

        save_path = Path(save_dir) / save_path

        save_path.parent.mkdir(exist_ok=True, parents=True)

        sitk_utils.write(sitk_image, save_path)


def mask_out_ct(ct_scan, ct_dir_path, masking_value):
    # Use, by priority, BODY, External or treatment table mask
    # to mask out unuseful values in the CT
    mask_exists = True
    negated_mask = False

    if (ct_dir_path / 'BODY.nrrd').exists():
        mask_path = ct_dir_path / 'BODY.nrrd'
    elif paths := io.find_paths_containing_pattern(ct_dir_path, "External*"):
        mask_path = paths[0]
    elif paths := io.find_paths_containing_pattern(ct_dir_path, "*_treatment_*"):
        mask_path = paths[0]
        # Use negated masking as we want to only mask out the table
        negated_mask = True
    else:
        mask_exists = False
        logger.info(f'No mask files found for {ct_dir_path}')

    if mask_exists:
        ct_mask = sitk_utils.load(mask_path)
        ct_scan = sitk_utils.apply_mask(ct_scan,
                                        ct_mask,
                                        masking_value=masking_value,
                                        set_same_origin=True,
                                        negated_mask=negated_mask)
    return ct_scan


def clamp_normalize(A, B, min_value, max_value):
    # Limits the lowest and highest HU unit
    A = torch.clamp(A, min_value, max_value)
    B = torch.clamp(B, min_value, max_value)

    # Normalize Hounsfield units to range [-1,1]
    A = min_max_normalize(A, min_value, max_value)
    B = min_max_normalize(B, min_value, max_value)

    # Add channel dimension (1 = grayscale)
    A = A.unsqueeze(0)
    B = B.unsqueeze(0)
    return A, B
