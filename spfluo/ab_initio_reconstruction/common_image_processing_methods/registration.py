import cc3d
import cupy as cp
import numpy as np
import SimpleITK as sitk
from cupyx.scipy.ndimage import fourier_shift as fourier_shift_cupy
from cupyx.scipy.ndimage import label as label_cupy
from numpy import pi
from scipy.ndimage import fourier_shift
from tqdm import tqdm


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


def translate_to_have_one_connected_component(
    im, t_min=-20, t_max=20, t_step=4, gpu=None
):
    ft = np.fft.fftn(im)
    if gpu == "cucim":
        ft_cupy = cp.array(ft)
    trans_vecs = np.arange(t_min, t_max, t_step)
    grid_trans_vec = np.array(
        np.meshgrid(trans_vecs, trans_vecs, trans_vecs)
    ).T.reshape((len(trans_vecs) ** 3, 3))
    number_connected_components = np.zeros(len(grid_trans_vec))
    t = 1.0
    for i, trans_vec in tqdm(
        enumerate(grid_trans_vec),
        leave=False,
        total=len(grid_trans_vec),
        desc="Translate to have one connected component",
    ):
        if gpu == "cucim":
            trans_vec = cp.array(trans_vec)
            ft_shifted = fourier_shift_cupy(ft_cupy, trans_vec)
            im_shifted = cp.fft.ifftn(ft_shifted)
            im_shifted_thresholded = cp.abs(im_shifted).real > t
            _, N = label_cupy(im_shifted_thresholded)
        else:
            ft_shifted = fourier_shift(ft, trans_vec)
            im_shifted = np.fft.ifftn(ft_shifted)
            im_shifted_thresholded = np.abs(im_shifted).real > t
            _, N = cc3d.connected_components(im_shifted_thresholded, return_N=True)
        number_connected_components[i] = N

    indicices_one_component = np.where(number_connected_components == 1)
    if len(indicices_one_component) == 1:
        indicices_one_component = np.where(
            number_connected_components == np.min(number_connected_components)
        )
    transvecs_one_components = grid_trans_vec[indicices_one_component]
    avg_transvec_one_component = transvecs_one_components[0]
    ft_shifted = fourier_shift(ft, avg_transvec_one_component)
    return np.abs(np.fft.ifftn(ft_shifted)).real
