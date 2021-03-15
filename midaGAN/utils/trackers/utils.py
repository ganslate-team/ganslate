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


def process_visuals_for_logging(visuals, single_example=False, grid_depth="full"):
    final_visuals_grids = []

    # When a list of visuals is given, process it recursively
    if isinstance(visuals, list):
        for single_visuals in visuals:
            single_visuals = process_visuals_for_logging(visuals, single_example, grid_depth)
            final_visuals_grids.extend(single_visuals)
        return final_visuals_grids

    # A single instance of visuals is a dict
    assert isinstance(visuals, dict)
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
