import torch
import torchvision


def visuals_to_combined_2d_grid(visuals):
    # if images are 3D (5D tensors)
    if len(list(visuals.values())[0].shape) == 5:  # TODO make nicer
        # Concatenate slices that are at the same level from different visuals along width.
        # Each tensor from visuals.values() is NxCxDxHxW, hence dim=4.
        combined_slices = torch.cat(tuple(visuals.values()), dim=4)
        # We plot a single volume from the batch
        combined_slices = combined_slices[0]
        # CxDxHxW -> DxCxHxW
        combined_slices = combined_slices.permute(1, 0, 2, 3)
        # Concatenate all combined slices along height to form a single 2d image.
        # Tensors in the tuple are CxHxW, hence dim=1
        combined_image = torch.cat(tuple(combined_slices), dim=1)
    else:
        # NxCxHxW
        combined_image = torch.cat(tuple(visuals.values()), dim=3)
        combined_image = combined_image[0]

    # Convert data range [-1,1] to [0,1]. Important when saving images.
    combined_image = (combined_image + 1) / 2

    # Name would be, e.g., "real_A-fake_B-rec_A-real_B-fake_A-rec_B"
    name = "-".join(visuals.keys())
    # NOTE: image format is CxHxW
    return {'name': name, 'image': combined_image}
