import copy
import torch
from midaGAN.utils import communication


def concat_batch_of_visuals_after_gather(visuals_list):
    # Gathering done only for rank 0 when DDP is ON
    visuals = visuals_list
    if torch.distributed.is_initialized() and communication.get_rank() == 0:
        visuals = visuals_list[0]
        for single_visuals in visuals_list[1:]:
            for key in single_visuals.keys():
                visuals[key] = torch.cat((visuals[key], single_visuals[key]), dim=0)
    return visuals


def convert_to_list_if_gather_did_not_occur(value):
    """When using `communication.gather()` the output is a list of gathered values from
    all proceses. However, that function gathers the values only on the process with rank 0,
    while the other processes have the same single value as before the gather. This function
    converts those values to a list so that the data type is identical to the rank 0 process,
    allowing more general implementation of logic that operates with these values.
    """
    # Gathering done only for rank 0 when DDP is ON
    if torch.distributed.is_initialized() and communication.get_rank() == 0:
        return value
    else:
        return [value]

def process_visuals_for_logging(visuals, visuals_config, single_example=False, grid_depth="full"):
    final_visuals_grids = []

    # When a list of visuals is given, process it recursively
    if isinstance(visuals, list):
        for single_visuals in visuals:
            single_visuals = process_visuals_for_logging(visuals, single_example, mid_slice_only)
            final_visuals_grids.extend(single_visuals)
        return final_visuals_grids

    # A single instance of visuals is a dict
    assert isinstance(visuals, dict)


    # Channel-wise splitting of multi-modality images into separate tensors
    visuals = _split_multimodal_visuals(visuals, visuals_config)
    
    # Make all visuals have the same number of channels, if different
    visuals = _make_all_visuals_channels_equal(visuals)

    visuals_list = list(visuals.values())
    is_three_dimensional = visuals_list[0].ndim == 5

    # Concatenate corresponding images/slices from different visuals along width.
    # Width, in case of 3D, is fourth dim, while in 2D it's the third.
    concat_dim = 4 if is_three_dimensional else 3
    batch_visuals_grids = torch.cat(tuple(visuals_list), dim=concat_dim)
    # When interested in logging a single example from the batch, go through it only
    if single_example:
        batch_visuals_grids = batch_visuals_grids[:1]

    for visuals_grid in batch_visuals_grids:
        if is_three_dimensional:
            # CxDxHxW -> DxCxHxW
            visuals_grid = visuals_grid.permute(1, 0, 2, 3)

            # Only mid slice from 3D image
            if mid_slice_only:
                mid_slice = visuals_grid.shape[0] // 2
                visuals_grid = visuals_grid[mid_slice]

            # Whole 3D image into a grid
            else:
                # Concatenate all combined slices along height to form a single 2D image.
                # Tensors in the tuple are CxHxW, hence dim=1
                visuals_grid = torch.cat(tuple(visuals_grid), dim=1)

        # Convert data range [-1,1] to [0,1]. Important when saving images.
        visuals_grid = (visuals_grid + 1) / 2

        # Name would be, e.g., "real_A-fake_B-rec_A-real_B-fake_A-rec_B"
        name = "-".join(visuals.keys())

        # Note that image format is CxHxW
        final_visuals_grids.append({'name': name, 'image': visuals_grid})

    return final_visuals_grids



def _split_multimodal_visuals(visuals, visuals_config):
    """
    Separate out multi-modality images from each tensor by splitting it channel-wise.
    The correct channel split for domains A and B should be provided in the visuals_config parameter.
    The visual names are updated as:  real_A  -->  real_A1 and real_A2 (in case when A contains 2 modalities).
    """
    # If visuals config is not provided, assume each visuals tensor contains a single modality, and do nothing
    if visuals_config is None:    
        return visuals

    # Split channels and update the visuals dict
    channel_splits_by_domain = {'A': visuals_config['channel_split_A'], 'B': visuals_config['channel_split_B']}
    visuals_copy = {}
    for visual_name in visuals.keys():                  # For each tensor in visuals
        for domain in channel_splits_by_domain.keys():  # For each domain (A and B)
            if domain in visual_name: 
                channel_split = tuple(channel_splits_by_domain[domain])
                assert sum(channel_split) == visuals[visual_name].shape[1], "Please specify channel-split correctly!"
                separated_modalities = torch.split(visuals[visual_name], channel_split, dim=1)
                visuals_copy.update({f"{visual_name}{i+1}": separated_modalities[i] for i in range(len(channel_split))})
    return visuals_copy


def _make_all_visuals_channels_equal(visuals):
    """
    Make #channels of all visuals equal to the largest #channels present.
    Especially useful in case of natural images when both RGB and grayscale image types are involved.
    Note:        This function is invoked after _split_multimodal_visuals(), 
                 and hence expects each tensor to contain just a single modality
    Limitation:  Every image modality must have #channels equal to either 1 or 3.                
    """
    supported_n_channels = (1,3) # Only 1 and 3 channels supported for each separate modality  
    max_n_channels = max([visual.shape[1] for visual in visuals.values()])
    min_n_channels = min([visual.shape[1] for visual in visuals.values()])
    
    # If all visuals have the same #channels, then do nothing 
    if max_n_channels == min_n_channels:
        return visuals

    # Else, proceed with the operation
    for visual_name in visuals.keys():
        n_channels = visuals[visual_name].shape[1]
        assert n_channels in supported_n_channels, "Every image modality must have #channels equal to either 1 or 3!"
        if n_channels < max_n_channels:
            n_repeats = max_n_channels // n_channels
            visuals[visual_name] = torch.repeat_interleave(visuals[visual_name], n_repeats, dim=1)

    return visuals