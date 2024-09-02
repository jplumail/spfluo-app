import numpy as np
import SimpleITK as sitk
from numpy import pi
from scipy.ndimage import fourier_shift, label
from skimage.filters import threshold_otsu
from skimage.measure import regionprops


def registration_exhaustive_search(
    fixed_image,
    moving_image,
    sample_per_axis=40,
    gradient_descent=False,
    threads=1,
):
    nb_dim = fixed_image.ndim
    fixed_image = sitk.GetImageFromArray(fixed_image)
    moving_image = sitk.GetImageFromArray(moving_image)
    trans = sitk.Euler3DTransform() if nb_dim == 3 else sitk.Euler2DTransform()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        trans,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    R = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)

    R.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    if not gradient_descent:
        if nb_dim == 2:
            R.SetOptimizerAsExhaustive([sample_per_axis, sample_per_axis, 0, 0])
            R.SetOptimizerScales(
                [2.0 * pi / sample_per_axis, 2.0 * pi / sample_per_axis, 1.0, 1.0]
            )
        else:
            R.SetOptimizerAsExhaustive(
                [sample_per_axis, sample_per_axis, sample_per_axis, 0, 0, 0]
            )
            R.SetOptimizerScales(
                [
                    2.0 * pi / sample_per_axis,
                    2.0 * pi / sample_per_axis,
                    2.0 * pi / sample_per_axis,
                    1.0,
                    1.0,
                    1.0,
                ]
            )
    if gradient_descent:
        R.SetOptimizerAsGradientDescent(0.1, 100)
    R.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    R.SetGlobalDefaultNumberOfThreads(threads)
    final_transform = R.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )

    moving_resampled = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )
    moving_resampled = sitk.GetArrayFromImage(moving_resampled)
    angle_X, angle_Y, angle_Z, _, _, _ = final_transform.GetParameters()
    return 180 * np.array([angle_X, angle_Y, angle_Z]) / np.pi, moving_resampled


def shift_registration_exhaustive_search(
    im1, im2, t_min=-20, t_max=20, t_step=4, fourier_space=False
):
    if fourier_space:
        ft1 = im1
        ft2 = im2
    else:
        ft1 = np.fft.fftn(im1)
        ft2 = np.fft.fftn(im2)
    trans_vecs = np.arange(t_min, t_max, t_step)
    grid_trans_vec = np.array(
        np.meshgrid(trans_vecs, trans_vecs, trans_vecs)
    ).T.reshape((len(trans_vecs) ** 3, 3))
    # print('len', len(grid_trans_vec))
    min_err = 10**20
    best_i = 0
    for i, trans_vec in enumerate(grid_trans_vec):
        ft2_shifted = fourier_shift(ft2, trans_vec)
        err = np.linalg.norm(ft2_shifted - ft1)
        if err < min_err:
            best_i = i
            min_err = err
    res = fourier_shift(ft2, grid_trans_vec[best_i])
    if not fourier_space:
        res = np.fft.ifftn(res)
    return grid_trans_vec[best_i], res.real.astype(im2.dtype)


def center_connected_component(image):
    # Step 1: Find connected components
    labeled_image, num_features = label(image > threshold_otsu(image))

    if num_features < 2:
        return image, np.array([0.0, 0.0, 0.0])  # Already has only one component

    # Step 2: Identify the largest connected component
    component_props = regionprops(labeled_image)
    largest_component = max(component_props, key=lambda prop: prop.area)

    # Step 3: Calculate translation
    center_of_mass = largest_component.centroid
    shift = np.asarray(image.shape) / 2 - np.asarray(center_of_mass)
    shift = np.rint(shift).astype(int)

    # Step 4: Apply translation
    translated_image = np.roll(image, tuple(shift), axis=tuple(range(image.ndim)))

    return translated_image, shift.astype(float)
