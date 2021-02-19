import torch


def visuals_to_combined_2d_grid(visuals, grid_depth='full'):
    assert isinstance(visuals, dict)
    visuals_list = list(visuals.values())

    if visuals_list[0].ndim == 5:  # TODO make nicer
        # Concatenate slices that are at the same level from different visuals along width.
        # Each tensor from visuals is NxCxDxHxW, hence dim=4.
        combined_slices = torch.cat(tuple(visuals_list), dim=4)
        # We plot a single volume from the batch
        combined_slices = combined_slices[0]
        # CxDxHxW -> DxCxHxW
        combined_slices = combined_slices.permute(1, 0, 2, 3)
        # Concatenate all combined slices along height to form a single 2d image.
        # Tensors in the tuple are CxHxW, hence dim=1
        if grid_depth == 'full':
            combined_image = torch.cat(tuple(combined_slices), dim=1)

        elif grid_depth == 'mid':
            mid_slice = combined_slices.shape[0] // 2
            combined_image = combined_slices[mid_slice]
    else:
        # NxCxHxW
        combined_image = torch.cat(tuple(visuals_list), dim=3)
        combined_image = combined_image[0]

    # Convert data range [-1,1] to [0,1]. Important when saving images.
    combined_image = (combined_image + 1) / 2

    # Name would be, e.g., "real_A-fake_B-rec_A-real_B-fake_A-rec_B"
    name = "-".join(visuals.keys())
    # NOTE: image format is CxHxW
    return {'name': name, 'image': combined_image}
