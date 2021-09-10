import torch
import wandb
from ganslate.utils import communication


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


def process_visuals_for_logging(conf, visuals, single_example=False, mid_slice_only=False):
    """Receives a dict of visuals and combines it for logging.
    `single_example` - selects only one example to log from the mini-batch of visuals.
    `mid_slice_only` - when visuals are 3D, setting it to True will log only the middle slice"""

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
    visuals = _split_multimodal_visuals(visuals, conf[conf.mode].logging.multi_modality_split)

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


def process_visuals_wandb_tensorboard(visuals, image_window=None, is_wandb=False):
    # Resursively process a list of visuals
    if isinstance(visuals, list):
        return [process_visuals_wandb_tensorboard(v, image_window, is_wandb) for v in visuals]

    name, image = visuals['name'], visuals['image']
    # CxHxW -> HxWxC
    image = image.permute(1, 2, 0)

    if image_window:
        image = image.clamp(image_window[0], image_window[1])
        image = (image - image_window[0]) / image_window[1] - image_window[0]

    if is_wandb:
        return wandb.Image(image.cpu().detach().numpy(), caption=name)
    return {"name": name, "image": image}


def _split_multimodal_visuals(visuals, multi_modality_split):
    """
    Separate out multi-modality images from each tensor by splitting it channel-wise.
    The correct channel split for domains A and B in `multi_modality_split` of the logging conf.
    The visual names are updated as: `real_A` -> `real_A1` and `real_A2`, if it has 2 modalities.
    """
    # If multi_modality_split not specified, assume each visuals are single modalities.
    if multi_modality_split is None:
        return visuals

    splitted_visuals = {}

    # For each tensor in visuals
    for name in visuals.keys():
        # Consider only those visuals whose names contain `_A` or `_B` (e.g. `real_A` or `fake_B`)
        if '_A' in name or '_B' in name:
            # For each domain (`A` and `B`)
            for domain in multi_modality_split:
                # Names of visuals ending with the domain name
                if name.endswith(domain):
                    channel_split = tuple(multi_modality_split[domain])
                    # Possible that the split is defined for only one of the two domains
                    if channel_split is None:
                        # Then just copy the visual
                        splitted_visuals[name] = visuals[name]
                        continue

                    # Num of channels in split need to correspond to the actual num of channels
                    if sum(channel_split) != visuals[name].shape[1]:
                        raise ValueError("Please specify channel-split correctly!")

                    # Split the modalities and assign them to visuals
                    splitted_modalities = torch.split(visuals[name], channel_split, dim=1)
                    for i in range(len(channel_split)):
                        splitted_visuals[f"{name}{i+1}"] = splitted_modalities[i]
        
        # No processing of visuals whose names do not contain `_A` or `_B` (for ex. masks with names `BODY` or `GTV`)
        else:
            splitted_visuals[name] = visuals[name]
 
    return splitted_visuals


def _make_all_visuals_channels_equal(visuals):
    """
    Make number of channels of all visuals equal to the largest number of channels among them.
    Especially useful in case of natural images when both RGB and grayscale image types are involved.
    Note:        This function is invoked after _split_multimodal_visuals(), 
                 and hence expects each tensor to contain just a single modality
    Limitation:  Every image modality must have #channels equal to either 1 or 3.                
    """

    max_n_channels = max([visual.shape[1] for visual in visuals.values()])
    min_n_channels = min([visual.shape[1] for visual in visuals.values()])
    # If all visuals have the same number of channels, return them
    if max_n_channels == min_n_channels:
        return visuals

    for name in visuals.keys():
        n_channels = visuals[name].shape[1]
        assert n_channels in (1, 3), "Every image must be either 1- or 3-channel image."
        if n_channels < max_n_channels:
            n_repeats = max_n_channels // n_channels
            visuals[name] = torch.repeat_interleave(visuals[name], n_repeats, dim=1)

    return visuals
