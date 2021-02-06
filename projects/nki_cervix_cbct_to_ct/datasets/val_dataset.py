import random
from dataclasses import dataclass, field
from pathlib import Path
# Config imports
from typing import Tuple, List

import midaGAN
import numpy as np
import torch
from midaGAN import configs
from midaGAN.data.utils import pad
from midaGAN.data.utils.body_mask import (apply_body_mask_and_bound,
                                          get_body_mask_and_bound)
from midaGAN.data.utils.fov_truncate import truncate_CBCT_based_on_fov
from midaGAN.data.utils.normalization import (min_max_denormalize,
                                              min_max_normalize)
from midaGAN.data.utils.registration_methods import (
    register_CT_to_CBCT, truncate_CT_to_scope_of_CBCT)
from midaGAN.data.utils.stochastic_focal_patching import \
    StochasticFocalPatchSampler
from midaGAN.utils import sitk_utils
from midaGAN.utils.io import load_json, make_recursive_dataset_of_files
from omegaconf import MISSING
from torch.utils.data import Dataset

DEBUG = False

import logging

logger = logging.getLogger(__name__)

EXTENSIONS = ['.nrrd']

# --------------------------- VALIDATION DATASET ---------------------------------------------
# --------------------------------------------------------------------------------------------

@dataclass
class CBCTtoCTValDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "CBCTtoCTValDataset"
    hounsfield_units_range: Tuple[int, int] = field(
        default_factory=lambda: (-1024, 2048))  #TODO: what should be the default range
    enable_masking: bool = True
    enable_bounding: bool = False
    cbct_mask_threshold: int = -700
    ct_mask_threshold: int = -300
    mask_labels: List[str] = field(
        default_factory=lambda: [])


class CBCTtoCTValDataset(Dataset):

    def __init__(self, conf):
        # self.paths = make_dataset_of_directories(conf.val.dataset.root, EXTENSIONS)
        self.root_path = Path(conf.val.dataset.root).resolve()
        self.mask_labels = conf.val.dataset.mask_labels

        self.paths = {}

        for patient in self.root_path.iterdir():

            # Sorted list of files is returned, pick the first CBCT volume.

            if (patient / "registered").is_dir():
                first_CBCT = (patient / "registered" / "target").with_suffix(".nrrd")
                dpCT = (patient / "registered" / "deformed").with_suffix(".nrrd")
                masks = {}
                for mask in self.mask_labels:
                    masks.update({mask: (patient / "registered" / mask).with_suffix(".nrrd")})

                self.paths[patient] = {"CT": dpCT, "CBCT": first_CBCT, "masks": masks}

        self.num_datapoints = len(self.paths)
        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.val.dataset.hounsfield_units_range

        self.apply_mask = conf.val.dataset.enable_masking
        self.apply_bound = conf.val.dataset.enable_bounding
        self.cbct_mask_threshold = conf.val.dataset.cbct_mask_threshold
        self.ct_mask_threshold = conf.val.dataset.ct_mask_threshold

    def __getitem__(self, index):
        patient_index = list(self.paths)[index]

        patient_dict = self.paths[patient_index]

        planning_CT_path = patient_dict["CT"]
        first_CBCT_path = patient_dict["CBCT"]
        mask_paths = patient_dict["masks"]

        # load nrrd as SimpleITK objects
        CT = sitk_utils.load(planning_CT_path)
        CBCT = sitk_utils.load(first_CBCT_path)

        # Load masks if they were found
        masks = {}
        if mask_paths:
            for label, mask_path in mask_paths.items():
                masks[label] = sitk_utils.load(mask_path)

        metadata = {
            'path': str(first_CBCT_path),
            'size': CBCT.GetSize(),
            'origin': CBCT.GetOrigin(),
            'spacing': CBCT.GetSpacing(),
            'direction': CBCT.GetDirection(),
            'dtype': sitk_utils.get_npy_dtype(CBCT)
        }

        CBCT = sitk_utils.get_npy(CBCT)
        CT = sitk_utils.get_npy(CT)
        masks = {k: sitk_utils.get_npy(v) for k,v in masks.items()}

        CT = torch.tensor(CT)
        CBCT = torch.tensor(CBCT)
        masks = {k: torch.tensor(v) for k,v in masks.items()}

        # Limits the lowest and highest HU unit
        CT = torch.clamp(CT, self.hu_min, self.hu_max)
        CBCT = torch.clamp(CBCT, self.hu_min, self.hu_max)
        # Normalize Hounsfield units to range [-1,1]
        CT = min_max_normalize(CT, self.hu_min, self.hu_max)
        CBCT = min_max_normalize(CBCT, self.hu_min, self.hu_max)
        # Add channel dimension (1 = grayscale)
        CT = CT.unsqueeze(0)
        CBCT = CBCT.unsqueeze(0)


        data_dict = {"A": CBCT, "B": CT, "metadata": metadata}
        if masks:
            data_dict.update({"masks": masks})

        return data_dict

    def denormalize(self, tensor):
        """Allows the Tester and Validator to calculate the metrics in
        the original range of values.
        """
        return min_max_denormalize(tensor.clone(), self.hu_min, self.hu_max)

    def save(self, tensor, metadata, output_dir):
        tensor = tensor.squeeze().cpu()
        tensor = min_max_denormalize(tensor, self.hu_min, self.hu_max)

        masking_value = torch.tensor(-1024, dtype=torch.float)

        if 'mask' in metadata or 'bounds' in metadata:

            full_tensor = torch.full(
                (metadata['size'][2], metadata['size'][1], metadata['size'][0]),
                masking_value,
                dtype=torch.float)

            if 'bounds' in metadata:
                bounds = metadata['bounds']
                full_tensor[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1],
                            bounds[2][0]:bounds[2][1]] = tensor

            if 'mask' in metadata:
                mask = torch.tensor(metadata['mask'], dtype=bool).squeeze()
                full_tensor = torch.where(mask, full_tensor, masking_value)

        else:
            full_tensor = tensor

        sitk_image = sitk_utils.tensor_to_sitk_image(full_tensor, metadata['origin'],
                                                     metadata['spacing'], metadata['direction'],
                                                     metadata['dtype'])

        # Dataset used has a directory per each datapoint, the name of each datapoint's dir is used to save the output
        datapoint_path = Path(str(metadata['path']))

        save_path = datapoint_path.relative_to(self.root_path)

        save_path = Path(output_dir) / save_path

        save_path.parent.mkdir(exist_ok=True, parents=True)

        sitk_utils.write(sitk_image, save_path)

    def __len__(self):
        return self.num_datapoints
