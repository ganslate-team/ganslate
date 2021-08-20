from pathlib import Path
from loguru import logger
import torch
from torch.utils.data import Dataset

import ganslate
from ganslate.utils import io, sitk_utils
from ganslate.data.utils.normalization import min_max_normalize, min_max_denormalize
from ganslate.data.utils.body_mask import apply_body_mask

# Config imports
from typing import Tuple
from dataclasses import dataclass
from omegaconf import MISSING
from ganslate import configs

from .common import (get_ct_body_mask_path, mask_out_ct, mask_out_registered_cbct_with_ct_mask,
                     clamp_normalize)


@dataclass
class CBCTtoCTValTestDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "CBCTtoCTValidationDataset"
    hounsfield_units_range: Tuple[int, int] = (-1000, 2000)


class CBCTtoCTValTestDataset(Dataset):

    def __init__(self, conf):
        self.root_path = Path(conf[conf.mode].dataset.root).resolve()
        self.pairs = sorted(io.make_recursive_dataset_of_directories(self.root_path, "nrrd"))
        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf[conf.mode].dataset.hounsfield_units_range

    def __getitem__(self, index):
        path_CBCT = self.pairs[index] / "target.nrrd"
        path_CT = self.pairs[index] / "deformed.nrrd"

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

        # Apply the mask
        CT = mask_out_ct(CT, path_CT.parent, self.hu_min)
        # Applying the CT mask on CBCT too since they're registered
        CBCT = mask_out_registered_cbct_with_ct_mask(CBCT, path_CT.parent, self.hu_min)

        # Mask out the body with morphological ops for consistency w training data
        CBCT = apply_body_mask(sitk_utils.get_npy(CBCT),
                               apply_mask=True,
                               masking_value=self.hu_min,
                               hu_threshold=-800)

        CT = apply_body_mask(sitk_utils.get_npy(CT),
                             apply_mask=True,
                             masking_value=self.hu_min,
                             hu_threshold=-600)

        CBCT = torch.tensor(CBCT)
        CT = torch.tensor(CT)

        CBCT, CT = clamp_normalize(CBCT, CT, self.hu_min, self.hu_max)

        out = {'A': CBCT, 'B': CT, "metadata": metadata}
        mask = get_ct_body_mask_path(path_CT.parent)
        if mask:
            mask = sitk_utils.get_tensor(sitk_utils.load(mask))
            mask = mask.unsqueeze(0)
            out["masks"] = {"BODY": mask}
        return out

    def __len__(self):
        return len(self.pairs)

    def denormalize(self, tensor):
        """Allows the Tester and Validator to calculate the metrics in
        the original range of values.
        """
        tensor = min_max_denormalize(tensor.clone(), self.hu_min, self.hu_max)

        # Offset tensor with hu_min to get CT number from HU
        # Useful for things like PSNR calculations
        return tensor - self.hu_min

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
