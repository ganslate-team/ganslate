from scipy import ndimage
import cv2
import numpy as np

def get_connected_components(binary_array: np.ndarray, structuring_element: np.ndarray=None) -> np.ndarray:
    """
    Returns a label map with a unique integer label for each connected geometrical object in the given binary array.
    Integer labels of components start from 1. Background is 0.
    """

    if not structuring_element:  # If not given, set 26-connected structure as default
        if binary_array.ndim == 3:
            cc_structure = np.array([
                                        [[1,1,1],
                                         [1,1,1],
                                         [1,1,1]],

                                        [[1,1,1],
                                         [1,1,1],
                                         [1,1,1]],

                                        [[1,1,1],
                                         [1,1,1],
                                         [1,1,1]]
                                       ])
        elif binary_array.ndim == 2:
            cc_structure = np.array(
                                        [[1,1,1],
                                         [1,1,1],
                                         [1,1,1]]

                                       )        
        else:
            raise NotImplementedError()
        # or alternatively, use the following function -
        # cc_structure = ndimage.generate_binary_structure(rank=3, connectivity=3)

    connected_component_array, num_connected_components = ndimage.label(binary_array, \
                                                                        structure=structuring_element)

    print("Number of connected components found:", num_connected_components)
    return connected_component_array



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
    label_counts = [np.sum(connected_components==label) for label in range(connected_components.max())]
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
        contours, hierarchy = cv2.findContours(binary_slice, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        # Get the largest contour based on its area and find the convex hull of it
        # Convex hull tutorial: https://www.learnopencv.com/convex-hull-using-opencv-in-python-and-c/
        max_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_cnt)

        # Project the hull onto the body_mask image, everything 
        # inside the hull is set to 1. 
        cv2.drawContours(body_mask[z], [hull], -1,  1, -1)

        # # Blurring the mask for smoother edges, this will also smooth out the jagged contours
        # # found in the sagittal view
        # body_mask[z] = cv2.GaussianBlur(body_mask[z], (5,5), 0)

    return body_mask, bound



def apply_body_mask_and_bound(array: np.ndarray, masking_value: int =-1024, \
                                apply_mask: bool =True, apply_bound: bool=True, HU_threshold: int =-300) -> np.ndarray:

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

    body_mask, ((z_max, z_min), \
        (y_max, y_min), (x_max, x_min)) = get_body_mask_and_bound(array, HU_threshold)
    
    # Apply mask to the image array 
    if apply_mask:
        array = np.where(body_mask, array, masking_value)

    # Index the array within the bounds and return cropped array
    if apply_bound:
        array = array[z_max:z_min, y_max: y_min, x_max: x_min]

    return array

    



