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


def convert_metrics_to_list_after_gather(metrics):
    # Gathering done only for rank 0 when DDP is ON
    if torch.distributed.is_initialized() and communication.get_rank() == 0:
        return metrics
    else:
        return [metrics]

def process_visuals_for_logging(visuals, visuals_config, single_example=False, grid_depth="full"):
    final_visuals_grids = []

    # When a list of visuals is given, process it recursively
    if isinstance(visuals, list):
        for single_visuals in visuals:
            single_visuals = process_visuals_for_logging(visuals, single_example, grid_depth)
            final_visuals_grids.extend(single_visuals)
        return final_visuals_grids

    # A single instance of visuals is a dict
    assert isinstance(visuals, dict)


    # Channel-wise splitting of multi-modality images
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

            # Whole 3D image into a grid
            if grid_depth == "full":
                # Concatenate all combined slices along height to form a single 2D image.
                # Tensors in the tuple are CxHxW, hence dim=1
                visuals_grid = torch.cat(tuple(visuals_grid), dim=1)

            # Only mid slice from 3D image
            elif grid_depth == "mid":
                mid_slice = visuals_grid.shape[0] // 2
                visuals_grid = visuals_grid[mid_slice]

        # Convert data range [-1,1] to [0,1]. Important when saving images.
        visuals_grid = (visuals_grid + 1) / 2

        # Name would be, e.g., "real_A-fake_B-rec_A-real_B-fake_A-rec_B"
        name = "-".join(visuals.keys())

        # Note that image format is CxHxW
        final_visuals_grids.append({'name': name, 'image': visuals_grid})

    return final_visuals_grids



def _split_multimodal_visuals(visuals, visuals_config):
    """
    TODO: Make the code cleaner
    Separate out multi-modality images from each tensor by splitting it channel-wise.
    The correct channel split for domains A and B should be provided in the visuals_config parameter.
    The visual names are updated as -- real_A  -->  real_A1 and real_A2 (in case when A contains 2 modalities).
    """
    # If visuals config is not provided, assume there are each tensor contains single modality, and do nothing 
    if visuals_config is None:    
        return visuals

    # Split channels and update the visuals dict
    channel_splits = {'A': visuals_config['channel_split_A'], 'B': visuals_config['channel_split_B']}
    visuals_copy = {}
    for visual_name in visuals.keys():
        for domain in channel_splits:
            if domain in visual_name:
                prev_split_idx = 0
                for count, split_idx in enumerate(channel_splits[domain]):
                    start_chan, end_chan = prev_split_idx, prev_split_idx + split_idx
                    visuals_copy[f"{visual_name}{count+1}"] = visuals[visual_name][:, start_chan:end_chan].clone()
                    prev_split_idx += split_idx
    return visuals_copy


def _make_all_visuals_channels_equal(visuals):
    """
    TODO: Better name for the function;  Make the code cleaner
    Make channels of all visuals equal to the largest number of channels.
    Especially useful in case of natural images when both RGB and grayscale image types are involved 
    Limitation:  The max channel number numst be divisible by channel number of all visuals. 
                 For example, 1-channel image can be made 3-channel, but 2-channel cannot be made 3-channel                   
    """
    max_channels = max([visual.shape[1] for visual in visuals.values()])
    min_channels = min([visual.shape[1] for visual in visuals.values()])
    
    # If all visuals have the same #channels, then do nothing 
    if max_channels == min_channels:
        return visuals

    # Else, proceed with the operation
    for visual_name in visuals.keys():
        n_channels = visuals[visual_name].shape[1]
        if n_channels < max_channels:
            assert max_channels % n_channels == 0  # max_channels should be divisible by n_channels
            n_repeats = max_channels // n_channels
            visuals[visual_name] = torch.repeat_interleave(visuals[visual_name], n_repeats, dim=1)

    return visuals