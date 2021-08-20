import SimpleITK as sitk
import numpy as np

from ganslate.utils import sitk_utils


def truncate_CBCT_based_on_fov(image: sitk.Image):
    """
    Truncates the CBCT to consider full FOV in the scans. First few and last few slices
    generally have small FOV that is around 25-50% of the axial slice. Ignore this 
    using simple value based filtering. 

    Parameters
    ---------------
    image: Input CBCT image to truncate. 

    Returns
    ----------------
    filtered_image: Truncated CBCT image
    """
    array = sitk.GetArrayFromImage(image)
    start_idx, end_idx = 0, array.shape[0]

    begin_truncate = False

    for idx, slice in enumerate(array):

        # Calculate the percentage FOV.
        # This should give an estimate of difference between
        # area of the z-axis rectangular slice and circle formed by
        # the FOV. Eg. 400x400 will have 160k area and if the FOV is
        # an end to end circle then it will have an area of 3.14*200*200
        percentage_fov = 1 - np.mean(slice == -1024)
        # As soon as the percentage of fov in the image
        # is above 75% of the image set the start index.
        if percentage_fov > 0.75 and start_idx == 0:
            start_idx = idx
            begin_truncate = True

        # Once the start index is set and the fov percentage
        # goes below 75% set the end index
        if begin_truncate and percentage_fov < 0.75:
            end_idx = idx - 1
            break

    image = sitk_utils.slice_image(image, start=(0, 0, start_idx), end=(-1, -1, end_idx))

    return image
