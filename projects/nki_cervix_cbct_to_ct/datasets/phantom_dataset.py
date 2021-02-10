import random
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
# Config imports
from typing import Dict, List, Tuple

import midaGAN
import numpy as np
import torch
from midaGAN import configs
from midaGAN.data.utils.fov_truncate import truncate_CBCT_based_on_fov
from midaGAN.data.utils.normalization import (min_max_denormalize, min_max_normalize)
from midaGAN.data.utils.ops import pad
from midaGAN.data.utils.registration_methods import (register_CT_to_CBCT,
                                                     truncate_CT_to_scope_of_CBCT)
from midaGAN.data.utils.stochastic_focal_patching import \
    StochasticFocalPatchSampler
from midaGAN.utils import sitk_utils
from midaGAN.utils.io import load_json, make_dataset_of_directories
from omegaconf import MISSING
from torch.utils.data import Dataset

DEBUG = False

import logging

logger = logging.getLogger(__name__)

EXTENSIONS = ['.nrrd']

# --------------------------- VALIDATION DATASET ---------------------------------------------
# --------------------------------------------------------------------------------------------


@dataclass
class ElektaPhantomDatasetConfig(configs.base.BaseDatasetConfig):
    name: str = "ElektaPhantomDataset"
    hounsfield_units_range: Tuple[int, int] = field(
        default_factory=lambda: (-1024, 2048))  #TODO: what should be the default range
    insert_values: Dict[str, int] = field(
        default_factory=lambda: {
            "Air": -1000,
            "LDPE": -100,
            "Polystyrene": -35,
            "Acrylic": 120,
            "Delrin": 340,
            "Teflon": 950,
            "plate": 0
        })


class ElektaPhantomDataset(Dataset):

    def __init__(self, conf):
        self.root_path = Path(conf.val.dataset.root).resolve()
        self.insert_values = conf.val.dataset.insert_values
        self.paths = {}

        for folder in self.root_path.iterdir():
            self.paths[folder] = {}

            for fn in folder.glob("*.nrrd"):
                self.paths[folder][fn.stem] = fn

            assert "CT" in self.paths[folder], \
                "CT.nrrd must be present to use the elekta phantom dataset"

            assert set(self.insert_values.keys()) == set([k for k in self.paths[folder].keys() if k != "CT"]), \
                "Masks for keys defined in 'insert_values' not found"

        self.num_datapoints = len(self.paths)

        # Min and max HU values for clipping and normalization
        self.hu_min, self.hu_max = conf.val.dataset.hounsfield_units_range

    def __getitem__(self, index):
        phantom_index = list(self.paths)[index]

        phantom_dict = self.paths[phantom_index]

        phantom_path = phantom_dict["CT"]
        mask_paths = {k: v for k, v in phantom_dict.items() if k != "CT"}

        # load nrrd as SimpleITK objects
        phantom = sitk_utils.load(phantom_path)
        
        insert_masks = {}
        for label, mask_path in mask_paths.items():
            insert_masks[label] = sitk_utils.load(mask_path)

        metadata = {
            'path': str(phantom_path),
            'size': phantom.GetSize(),
            'origin': phantom.GetOrigin(),
            'spacing': phantom.GetSpacing(),
            'direction': phantom.GetDirection(),
            'dtype': sitk_utils.get_npy_dtype(phantom)
        }

        phantom = sitk_utils.get_npy(phantom)
        insert_masks = {k: sitk_utils.get_npy(v) for k, v in insert_masks.items()}

        # Limit phantom to z where plate for Image Quality assessment is present
        z_range = np.nonzero(insert_masks["plate"])[0]
        z_min, z_max = z_range.min(), z_range.max()

        phantom = phantom[z_min:z_max]
        insert_masks = {k: v[z_min:z_max] for k, v in insert_masks.items()}

        target_phantom = np.full(phantom.shape, self.hu_min, dtype=phantom.dtype)
        for label, mask in insert_masks.items():
            target_phantom = np.where(mask, self.insert_values[label], target_phantom)

        phantom = torch.tensor(phantom)
        target_phantom = torch.tensor(target_phantom)
        insert_masks = {k: torch.tensor(v) for k, v in insert_masks.items()}

        # Limits the lowest and highest HU unit
        phantom = torch.clamp(phantom, self.hu_min, self.hu_max)
        target_phantom = torch.clamp(target_phantom, self.hu_min, self.hu_max)

        # Normalize Hounsfield units to range [-1,1]
        phantom = min_max_normalize(phantom, self.hu_min, self.hu_max)
        target_phantom = min_max_normalize(target_phantom, self.hu_min, self.hu_max)

        # Add channel dimension (1 = grayscale)
        phantom = phantom.unsqueeze(0)
        target_phantom = target_phantom.unsqueeze(0)
        insert_masks = {k: v.unsqueeze(0) for k, v in insert_masks.items()}

        return {"A": phantom, "B": target_phantom, "masks": insert_masks, "metadata": metadata}

    def denormalize(self, tensor):
        """Allows the Tester and Validator to calculate the metrics in
        the original range of values.
        """
        tensor = min_max_denormalize(tensor.clone(), self.hu_min, self.hu_max)

        # Offset tensor with hu_min to get CT number from HU
        # Useful for things like PSNR calculations
        return tensor - self.hu_min

    def save(self, tensor, save_dir, metadata=None):
        tensor = tensor.squeeze().cpu()
        tensor = self.denormalize(tensor)

        if metadata:
            sitk_image = sitk_utils.tensor_to_sitk_image(tensor, metadata['origin'],
                                                         metadata['spacing'], metadata['direction'],
                                                         metadata['dtype'])
            datapoint_path = Path(str(metadata['path']))
            save_path = datapoint_path.relative_to(self.root_path)

        else:
            sitk_image = sitk_utils.tensor_to_sitk_image(tensor)
            save_path = f'image_{date.today().strftime("%b-%d-%Y")}.nrrd'
        # Dataset used has a directory per each datapoint,
        # the name of each datapoint's dir is used to save the output
        save_path = Path(save_dir) / save_path
        save_path.parent.mkdir(exist_ok=True, parents=True)
        sitk_utils.write(sitk_image, save_path)

    def __len__(self):
        return self.num_datapoints
