import torch
import torchvision

def visuals_to_combined_2d_grid(visuals):
    # if images are 3D (5D tensors)
    if len(list(visuals.values())[0].shape) == 5: # TODO make nicer
        # Concatenate slices that are at the same level from different visuals along 
        # width (each tensor from visuals.values() is NxCxDxHxW, hence dim=4)
        combined_slices = torch.cat(tuple(visuals.values()), dim=4) 
        combined_slices = combined_slices[0] # we plot a single volume from the batch
        combined_slices = combined_slices.permute(1,0,2,3) # CxDxHxW -> DxCxHxW
        # Concatenate all combined slices along height to form a single 2d image (tensors in tuple are CxHxW, hence dim=1)
        combined_image = torch.cat(tuple(combined_slices), dim=1) 
    else:
        # NxCxHxW
        combined_image = torch.cat(tuple(visuals.values()), dim=3)
        combined_image = combined_image[0]

    combined_image = (combined_image + 1) / 2 # [-1,1] -> [0,1]. Data range important when saving images.
    name = "-".join(visuals.keys()) # e.g. "real_A-fake_B-rec_A-real_B-fake_A-rec_B"
    return {'name': name, 'image': combined_image} # NOTE: image format is CxHxW   
