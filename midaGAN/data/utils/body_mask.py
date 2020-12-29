from scipy import ndimage
import cv2
import numpy as np

import logging
logger = logging.getLogger(__name__)


def get_connected_components(binary_array: np.ndarray,
                             structuring_element: np.ndarray = None) -> np.ndarray:
    """
    Returns a label map with a unique integer label for each connected geometrical object in the given binary array.
    Integer labels of components start from 1. Background is 0.
    """

    if not structuring_element:  # If not given, set 26-connected structure as default
        if binary_array.ndim == 3:
            cc_structure = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                     [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                     [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
        elif binary_array.ndim == 2:
            cc_structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        else:
            raise NotImplementedError()
        # or alternatively, use the following function -
        # cc_structure = ndimage.generate_binary_structure(rank=3, connectivity=3)

    connected_component_array, num_connected_components = ndimage.label(binary_array, \
                                                                        structure=structuring_element)

    return connected_component_array


def smooth_contour_points(contour: np.ndarray, radius: int = 3, sigma: int = 10) -> np.ndarray:
    """
    Function that smooths contour points using the approach from 
    https://stackoverflow.com/a/37536310
    
    Simple explanation: Convolve 1D gaussian filter over the points to smoothen the curve
    """
    neighbourhood = 2 * radius + 1

    # Contour length is the total number of points + extra points
    # to ensure circularity.
    contour_length = len(contour) + 2 * radius
    # Last group of points.
    offset = (len(contour) - radius)

    x_filtered, y_filtered = [], []

    for idx in range(contour_length):
        x_filtered.append(contour[(offset + idx) \
                                          % len(contour)][0][0])

        y_filtered.append(contour[(offset + idx) \
                                          % len(contour)][0][1])

    # Gaussian blur from opencv is basically applying gaussian convolution
    # filter over these points.
    x_smooth = cv2.GaussianBlur(np.array(x_filtered), (radius, 1), sigma)
    y_smooth = cv2.GaussianBlur(np.array(y_filtered), (radius, 1), sigma)

    # Add smoothened point for
    smooth_contours = []
    for idx, (x, y) in enumerate(zip(x_smooth, y_smooth)):
        if idx < len(contour) + radius:
            smooth_contours.append(np.array([x, y]))

    return np.array(smooth_contours)


def get_body_mask_and_bound(image: np.ndarray, HU_threshold: int) -> np.ndarray:
    """
    Function that gets a mask around the patient body and returns a 3D bound

    Parameters
    -------------
    image: Numpy array to get the mask and bound from
    HU_threshold: Set threshold to binarize image


    Returns
    -------------
    body_mask: Numpy array with same shape as input image as a body mask
    bound: Bounds around the largest component in 3D. This is in
    the ((z_min, z_max), (y_min, y_max), (x_min, x_max)) format
    """

    binarized_image = np.uint8(image >= HU_threshold)

    body_mask = np.zeros(image.shape)

    connected_components = get_connected_components(binarized_image)

    # Get counts for each component in the connected component analysis
    label_counts = [
        np.sum(connected_components == label) for label in range(1,
                                                                 connected_components.max() + 1)
    ]
    max_label = np.argmax(label_counts) + 1

    # Image with largest component binary mask
    binarized_image = connected_components == max_label

    # Get coordinates where a label (1) is present
    label_coordinates = np.nonzero(binarized_image)

    # Get bound of the largest possible voxel range in the binary mask
    bound = [(np.min(coord), np.max(coord)) for coord in label_coordinates]

    for z in range(binarized_image.shape[0]):

        binary_slice = np.uint8(binarized_image[z])

        # Find contours for each binary slice
        try:
            contours, hierarchy = cv2.findContours(binary_slice, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            # Get the largest contour based on its area
            largest_contour = max(contours, key=cv2.contourArea)

        except:
            logger.debug(
                "OpenCV could not find contours: Most likely this is a completely black image")
            continue

        # Smooth contour so that surface irregularities are removed better
        smoothed_contour = smooth_contour_points(largest_contour)

        # Project the points onto the body_mask image, everything
        # inside the points is set to 1.
        cv2.drawContours(body_mask[z], [smoothed_contour], -1, 1, -1)

    return body_mask, bound



def apply_body_mask_and_bound(array: np.ndarray, masking_value: int =-1024, \
                                apply_mask: bool =False, apply_bound: bool=False, HU_threshold: int =-300) -> np.ndarray:
    """
    Function to apply mask based filtering and bound the array
    
    Parameters
    ------------------
    array: Input array to bound and mask
    masking_value: Value to apply outside the mask
    apply_mask: Set to True to apply mask
    apply_bound: Set to True to apply bound
    HU_threshold: Threshold to apply for binarization of the image. 

    Returns
    ------------------
    array: Output array that will be masked with the masking_value outside
    the patient body and will be cropped to fit the bounds.

    """
    if not (apply_mask or apply_bound):
        return array

    body_mask, ((z_max, z_min), \
        (y_max, y_min), (x_max, x_min)) = get_body_mask_and_bound(array, HU_threshold)

    # Apply mask to the image array
    if apply_mask:
        array = np.where(body_mask, array, masking_value)

    # Index the array within the bounds and return cropped array
    if apply_bound:
        array = array[z_max:z_min, y_max:y_min, x_max:x_min]

    return array
