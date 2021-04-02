import SimpleITK as sitk
from numpy import mean
from itertools import product
from loguru import logger
import os
import traceback

REGISTRATION_MAP = {"Affine": sitk.AffineTransform(3), "Rigid": sitk.Euler3DTransform()}


def truncate_CT_to_scope_of_CBCT(CT, CBCT):
    """CBCT scans are usually focused on smaller part of the body compared to what is
    represented in CT. This function limits a CT to the part of the body that is found
    in CBCT by registering CT to the CBCT and cropping it to the relevant scope.
    https://discourse.itk.org/t/registration-to-get-rid-of-slices-containing-part-of-the-body-not-found-in-other-scan/3313/4
    """
    try:
        registration_transform = get_registration_transform(fixed_image=CBCT, moving_image=CT)
    except BaseException as e:
        logger.error(f"Registration failed with error: {traceback.print_exc()}")
        return CT

    # Start and end positions of CBCT volume
    start_position = [0, 0, 0]
    end_position = [point - 1 for point in CBCT.GetSize()]
    # Get all corner points of the CBCT volume
    corners = list(product(*zip(start_position, end_position)))

    # Transform all corner points from index to physical location
    physical_corners = [CBCT.TransformIndexToPhysicalPoint(corner) for corner in corners]

    # Find where is the scope of CBCT located in CT by using the data of how CT was registered to it
    transformed_corners = [
        registration_transform.TransformPoint(corner) for corner in physical_corners
    ]

    # Transform all corners points from physical to index location in CT
    final_corners = [CT.TransformPhysicalPointToIndex(corner) for corner in transformed_corners]

    # Get z-axis (slice) index of each corner and sort them.
    # The first four elements of it are the slice indices of the bottom of the volume,
    # while the other four are the slice indices of the top of the volume.
    z_corners = sorted([xyz[2] for xyz in final_corners])

    # The registered image can be sloped it in regards to z-axis,
    # so its corners might not lay in the same slice.
    # Averaging them is a way to decide at which bottom and top slice the CT should be truncated.
    start_slice = int(round(mean(z_corners[:4])))
    end_slice = int(round(mean(z_corners[4:])))
    # When the registration fails, just return the original CT. Happens infrequently.
    if start_slice < 0:
        logger.info("Registration failed as the at least one corner is below 0 in one of the axes."
                    " Passing the whole CT volume.")
        return CT
    return CT[:, :, start_slice:end_slice]


def register_CT_to_CBCT(CT, CBCT, registration_type="Rigid"):
    # TODO: make it more general sounding, except assumes CT is bigger
    try:
        registration_transform = get_registration_transform(fixed_image=CBCT,
                                                            moving_image=CT,
                                                            registration_type=registration_type)
        return sitk.Resample(CT, CBCT, registration_transform, sitk.sitkLinear, -1024,
                             CT.GetPixelID())

    except BaseException as e:
        logger.debug(f"Registration failed with error: {traceback.print_exc()}")

        # If Registration failed, then center crop CT: Last resort
        start_point = [(v1 - v2) // 2 for v1, v2 in zip(CT.GetSize(), CBCT.GetSize())]
        end_point = [v1 + v2 for v1, v2 in zip(start_point, CBCT.GetSize())]
        CT = CT[start_point[0]:end_point[0], start_point[1]:end_point[1],
                start_point[2]:end_point[2]]
        return CT


def get_registration_transform(fixed_image, moving_image, registration_type="Rigid"):
    """Performs the registration and returns a SimpleITK's `Transform` class which can be
    used to resample an image so that it is registered to another one. However, in our code
    should be truncated so that it contains only the part of the body that is 
    found in the `fixed_image`.
    Registration parameters are hardcoded and picked for the specific 
    task of  CBCT to CT translation. 
    TODO: consider making the adjustable in config.
    
    Parameters:
    ------------------------
    fixed_image:
    moving_image: 
    registration_type: Type of transformation to be applied to the moving image
    http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/22_Transforms.html
    
    """

    # Get seed from environment variable if set for registration randomness
    if 'PYTHONHASHSEED' in os.environ:
        seed = int(os.environ.get('PYTHONHASHSEED'))
    else:
        seed = sitk.sitkWallClock

    # SimpleITK registration's supported pixel types are sitkFloat32 and sitkFloat64
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=200)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)

    registration_method.SetMetricSamplingPercentage(0.01, seed)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,
                                                      numberOfIterations=200,
                                                      convergenceMinimumValue=1e-6,
                                                      convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if registration_type in REGISTRATION_MAP:
        registration_transform = REGISTRATION_MAP[registration_type]
    else:
        logger.warning("Unsupported transform provided, falling back to Rigid transformation")
        registration_transform = REGISTRATION_MAP["Rigid"]

    # Align the centers of the two volumes and set the
    # center of rotation to the center of the fixed image
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, registration_transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetNumberOfThreads(1)

    final_transform = registration_method.Execute(fixed_image, moving_image)
    return final_transform
