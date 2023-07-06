import time

import SimpleITK as sitk
from numpy import pi
import os
from tqdm import tqdm
import numpy as np
from scipy.ndimage import fourier_shift
import cupy as cp
from cupyx.scipy.ndimage import label as label_cupy, fourier_shift as fourier_shift_cupy
import cc3d
from ..manage_files.read_save_files import read_image, save


def registration_exhaustive_search(
    fixed_image,
    moving_image,
    output_dir,
    output_name,
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
    if output_dir is not None and output_name is not None:
        sitk.WriteImage(moving_resampled, os.path.join(output_dir, f"{output_name}"))
        im = read_image(f"{output_dir}/{output_name}")
        save(f"{output_dir}/{output_name}", im)
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


if __name__ == "__main__":
    from skimage.registration import phase_cross_correlation

    #
    from skimage.metrics import structural_similarity as ssim

    pth = "/home/eloy/Documents/article_reconstruction_micro_fluo/TCI23"
    im_to_register = ["AMPA_5p_lambda1e-3"]

    gts = ["recepteurs_AMPA"]

    for i in range(len(gts)):
        pth_im = f"{pth}/{im_to_register[i]}.tif"
        pth_gt = f"{PTH_GT}/{gts[i]}.tif"
        registration_exhaustive_search(
            pth_gt,
            pth_im,
            pth,
            f"{im_to_register[i]}_registered",
            3,
            sample_per_axis=40,
            gradient_descent=False,
        )
    1 / 0

    pth = "/home/eloy/Documents/stage_reconstruction_spfluo/results_hpc/recepteurs_AMPA/test_nb_views_anis/test_with_pl"
    nb_viewss = [50, 60, 80]
    sigs_z = [5, 10, 15, 20]
    for nb_views in nb_viewss:
        for sig_z in sigs_z:
            pth2 = f"{pth}/nb_views_{nb_views}/sig_z_{sig_z}/test_0/recons.tif"
            im = read_image(pth2)
            translated = translate_to_have_one_connected_component(im)
            pth_save = (
                f"{pth}/nb_views_{nb_views}/sig_z_{sig_z}/test_0/recons_translated.tif"
            )
            save(pth_save, translated)

    1 / 0
    im1 = np.random.random((50, 50, 50))
    im2 = np.random.random((50, 50, 50))
    t = time.time()
    for i in range(100):
        phase_cross_correlation(im1, im2, upsample_factor=1)
    # print('temps registration', time.time()-t)
    1 / 0

    pth = "/home/eloy/Documents/stage_reconstruction_spfluo/results/recepteurs_AMPA/evth_unknwon/test_0"
    pth_recons = f"{pth}/recons.tif"
    part_name = "recepteurs_AMPA"
    """
    recons = read_image(pth_recons)
    
    gt = read_image(f'/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{part_name}.tif')
    _, registered = shift_registration_exhaustive_search(gt, recons)
    save(f'{pth}/recons_registered.tif', registered)
    """
    registration_exhaustive_search(
        f"/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{part_name}.tif",
        f"{pth}/recons_registered.tif",
        pth,
        "recons_registered_trans_and_rot.tif",
        3,
        sample_per_axis=40,
        gradient_descent=False,
    )

    1 / 0
    pth = f"/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine/selected_data/results/c1_2_views/coeff_kernel_rot_50_axes_50_N_sample_512gaussian_kernel_rot"
    pth_im = f"{pth}/recons.tif"
    pth_fixed_im = "/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine/selected_data/results/c1/top_view.tif"
    output_name = "recons_registered"
    registration_exhaustive_search(
        pth_fixed_im,
        pth_im,
        pth,
        output_name,
        3,
        sample_per_axis=40,
        gradient_descent=False,
    )

    1 / 0

    fold = "/home/eloy/Documents/stage_reconstruction_spfluo/results_scipion/tomographic_reconstruction"
    part_names = ["recepteurs_AMPA", "HIV-1-Vaccine_prep", "clathrine", "emd_0680"]
    for part_name in part_names:
        recons = read_image(f"{fold}/{part_name}/nb_views_10/recons.tif")
        gt_path = f"/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{part_name}.tif"
        fold_part = f"{fold}/{part_name}/nb_views_10"
        registration_exhaustive_search(
            gt_path,
            f"{fold_part}/recons.tif",
            fold_part,
            "recons_registered",
            3,
            sample_per_axis=40,
            gradient_descent=False,
        )
        gt = read_image(gt_path)
        recons_registered = read_image(f"{fold_part}/recons_registered.tif")
        ssim_val = ssim(gt, recons_registered)
        # print(f'ssim of {part_name}', ssim_val)

    """
    pth_results = f'{PATH_REAL_DATA}/Data_marine_raw_prep/c1_results/nb_views_60/test_0/set_0'
    pth1 = f'{pth_results}/intermediar_results/recons_epoch_16.tif'

    im = read_image(pth1)
    im_shifted = translate_to_have_one_connected_component(im, -40, 40, 4)
    save(f'{pth_results}/recons_shifted.tif',im_shifted)
    """
    """
    gt_name = 'recepteurs_AMPA'
    gt_path = '/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/recepteurs_AMPA.tif'
    path_init_vol = '/home/eloy/Documents/stage_reconstruction_spfluo/results/recepteurs_AMPA/sig_z_5/test_0_conv_ZXZ/recons_registered.tif'
    path_reg_vol = '/home/eloy/Documents/stage_reconstruction_spfluo/results/recepteurs_AMPA/sig_z_5/test_0_conv_ZXZ'
    
    registration_exhaustive_search(gt_path, path_init_vol, path_reg_vol,
                                   'recons_registered2', 3,
                                   sample_per_axis=50)
    
    from time import time
    from skimage import io
#
#
    pth1 = f'{PATH_REAL_DATA}/picked_centrioles_preprocessed_results_0/recons.tif'
    pth2 = f'{PATH_REAL_DATA}/picked_centrioles_preprocessed_results_1/intermediar_results/recons_epoch_6.tif'
    recons_1 = read_image(pth1)
    recons_2 = read_image(pth2)
    _, registered_recons_2 = shift_registration_exhaustive_search(recons_1, recons_2, -20, 20, 4)
    #print('shift registration finished')
    save(f'{PATH_REAL_DATA}/picked_centrioles_preprocessed_results_1/recons_shift_registered.tif', registered_recons_2)
    registration_exhaustive_search(pth1, pth2, f'{PATH_REAL_DATA}/picked_centrioles_preprocessed_results_1', 'recons_registered', 3)


    
    recons = io.imread(pth_recons_real_data)

    shift = np.array([0,0,7])
    # Phase-shift
    t = time()
    fourier_shifted_image = fourier_shift(np.fft.fftn(recons), shift)

    shifted_image = np.fft.ifftn(fourier_shifted_image)
    #print('time shift', time() - t)
    #print(shifted_image)
    pth_shifted = f'{pth_folder_res}/recons_shifted.tif'
    save(pth_shifted, shifted_image)
    """
