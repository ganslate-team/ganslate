import logging

import torch
from midaGAN.data.utils.normalization import min_max_normalize
from midaGAN.utils import io, sitk_utils

logger = logging.getLogger(__name__)


def mask_out_ct(ct_scan, ct_dir_path, masking_value):
    # Use, by priority, BODY, External or treatment table mask
    # to mask out unuseful values in the CT
    mask_path = None
    negated_mask = False

    if path := get_ct_body_mask_path(ct_dir_path):
        mask_path = path
    elif paths := io.find_paths_containing_pattern(ct_dir_path, "*_treatment_*"):
        mask_path = paths[0]
        # Use negated masking as we want to only mask out the table
        negated_mask = True
    else:
        logger.info(f'No mask files found for {ct_dir_path}')

    if mask_path is not None:
        ct_mask = sitk_utils.load(mask_path)
        ct_scan = sitk_utils.apply_mask(ct_scan,
                                        ct_mask,
                                        masking_value=masking_value,
                                        set_same_origin=True,
                                        negated_mask=negated_mask)

    return ct_scan


def mask_out_registered_cbct_with_ct_mask(cbct_scan, ct_dir_path, masking_value):
    if path := get_ct_body_mask_path(ct_dir_path):
        ct_mask = sitk_utils.load(path)
        cbct_scan = sitk_utils.apply_mask(cbct_scan,
                                          ct_mask,
                                          masking_value=masking_value,
                                          set_same_origin=True)

    return cbct_scan


def get_ct_body_mask_path(ct_dir_path):
    if (ct_dir_path / 'BODY.nrrd').exists():
        return ct_dir_path / 'BODY.nrrd'
    elif paths := io.find_paths_containing_pattern(ct_dir_path, "External*"):
        return paths[0]
    return None


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
