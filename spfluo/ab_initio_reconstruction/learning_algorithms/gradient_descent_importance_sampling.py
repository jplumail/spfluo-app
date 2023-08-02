import copy
import json
import os
import shutil
from typing import Tuple

import array_api_compat
import cupy as cp
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R
from skimage import io
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm

from spfluo.ab_initio_reconstruction.volume_representation.pixel_representation import (
    Fourier_pixel_representation,
)
from spfluo.utils._array import Array
from spfluo.utils.transform import get_transform_matrix
from spfluo.utils.volume import fourier_shift, phase_cross_correlation

from ..common_image_processing_methods.others import normalize, stopping_criteria
from ..common_image_processing_methods.rotation_translation import (
    conversion_2_first_eulers_angles_cartesian,
    rotation,
)
from ..manage_files.read_save_files import make_dir, save, write_array_csv
from ..volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import (  # noqa: E501
    one_d_gaussian,
)


def gd_importance_sampling_3d(
    volume_representation: Fourier_pixel_representation,
    uniform_sphere_discretization,
    true_trans_vecs,
    views,
    imp_distrs_axes,
    imp_distrs_rot,
    unif_prop,
    unif_prop_min,
    params_learning_alg,
    output_dir,
    ground_truth=None,
    file_names=None,
    folder_views_selected=None,
    gpu=None,
):
    unif_prop_axes, unif_prop_rot = unif_prop
    epoch_length = (
        params_learning_alg.epoch_length
        if params_learning_alg.epoch_length is not None
        else len(views)
    )
    bs = params_learning_alg.batch_size

    if folder_views_selected is None:
        folder_views_selected = f"{output_dir}/views_selected"
        make_dir(folder_views_selected)

    make_dir(output_dir)
    # print('number of views', len(views))
    (thetas, phis, psis), _ = uniform_sphere_discretization
    x, y, z = conversion_2_first_eulers_angles_cartesian(thetas, phis)
    axes = np.array([x, y, z])
    M_axes = len(thetas)
    M_rot = len(psis)
    imp_distrs_axes_recorded = []
    imp_distrs_rot_recorded = []
    recorded_energies = []
    energies_each_view = [[] for _ in range(len(views))]
    itr = 0

    recorded_shifts = [[] for _ in range(len(views))]
    ssims = []
    sub_dir = os.path.join(output_dir, "intermediar_results")
    make_dir(sub_dir)
    ests_poses = []
    nb_step_of_supress = 0
    pbar = tqdm(total=params_learning_alg.N_iter_max, leave=False, desc="energy : +inf")
    views = views.astype(params_learning_alg.dtype)
    N_axes, N_rot = params_learning_alg.N_axes, params_learning_alg.N_rot
    while itr < params_learning_alg.N_iter_max and (
        not stopping_criteria(recorded_energies, params_learning_alg.eps)
    ):
        # print(f'nb views epoch {itr} : ', len(views))

        nb_views = len(views)
        itr += 1
        total_energy = 0
        estimated_poses_iter = np.zeros((nb_views, 6))

        if params_learning_alg.random_sampling:
            # Weighted-random sampling
            weights = np.ones((nb_views,)) / nb_views
            last_energies = np.array(
                [
                    energies_each_view[v][-1]
                    if len(energies_each_view[v]) > 0
                    else np.inf
                    for v in range(len(views))
                ]
            )
            m = last_energies.max()
            if np.isinf(m):
                weights = np.array([1.0 if np.isinf(e) else 0.0 for e in last_energies])
                weights /= weights.sum()
            else:
                last_energies_centered = last_energies - m
                weights = (
                    np.exp(params_learning_alg.beta_sampling * last_energies_centered)
                    / np.exp(
                        params_learning_alg.beta_sampling * last_energies_centered
                    ).sum()
                )  # softmax
            chosen_views = np.random.choice(nb_views, size=epoch_length, p=weights)
        else:
            chosen_views = np.arange(epoch_length) % len(views)
            np.random.shuffle(chosen_views)
        batches = [
            chosen_views[i * bs : (i + 1) * bs] for i in range(epoch_length // bs)
        ]
        last_batch_size = epoch_length % bs
        if last_batch_size > 0:
            batches += [chosen_views[-last_batch_size:]]

        pbar2 = tqdm(batches, leave=False)
        for batch in pbar2:
            gradients_batch = []
            energies_batch = []
            for v in batch:
                # Pick a subset of the discretization of SO(3) based on
                # importance distributions
                indices_axes = np.random.choice(
                    range(M_axes), p=imp_distrs_axes[v], size=N_axes
                )
                indices_rot = np.random.choice(
                    range(M_rot), p=imp_distrs_rot[v], size=N_rot
                )
                rot_vecs = np.stack(
                    np.broadcast_arrays(
                        thetas[indices_axes][:, None],
                        phis[indices_axes][:, None],
                        psis[indices_rot],
                    ),
                    axis=-1,
                )  # the euler angles, shape N_axes x N_rot x 3
                image_shape = views[0].shape
                # turn the euler angles into transform matrices, shape N_axes*N_rot x 3
                transforms = get_transform_matrix(
                    shape=image_shape,
                    euler_angles=rot_vecs.reshape(-1, 3),
                    translation=np.zeros((N_axes * N_rot, 3)),
                    convention=params_learning_alg.convention,
                    degrees=True,
                )
                # transforms represents transformation from reference volume
                # to the views

                # inverse_transforms represents transformation from the views
                # to the reference volume
                inverse_transforms = np.linalg.inv(transforms)

                view = views[v]
                if gpu == "pytorch":
                    inverse_transforms, view = map(
                        lambda x: torch.as_tensor(
                            x.astype(params_learning_alg.dtype), device="cuda"
                        ),
                        [inverse_transforms, view],
                    )
                elif gpu == "cucim":
                    inverse_transforms, view = map(
                        lambda x: cp.asarray(x.astype(params_learning_alg.dtype)),
                        [inverse_transforms, view],
                    )

                # Compute shifts
                # views are transformed back to the reference volume
                # then phase cross correlation is computed to get the shifts
                (
                    shifts,
                    psf_inverse_transformed_fft,
                    view_inverse_transformed_fft,
                ) = compute_shifts(
                    volume_representation.volume_fourier,
                    volume_representation.psf,
                    inverse_transforms,
                    view,
                    interp_order=1,
                )

                # Shift the view
                view_inverse_transformed_shifted_fft = fourier_shift(
                    view_inverse_transformed_fft, shifts
                )

                # Compute the energy associated with each transformation
                energies = compute_energy(
                    volume_representation.volume_fourier,
                    psf_inverse_transformed_fft,
                    view_inverse_transformed_shifted_fft,
                )

                # Recover the pose of the view

                # To go from view to the reference volume, we did:
                # an inverse transform, which is a pure rotation lets call it R^-1
                # then a shift, T_{t}

                # So, volume = (T_{t} o R^-1)(view)
                # If we invert, (R o T_{-t})(volume) = view
                # Equivalently, (T_{- R^-1 x t} o R)(volume) = view

                # The associated pose with the view is therefore
                rot = R.from_euler(
                    params_learning_alg.convention,
                    rot_vecs.reshape(-1, 3),
                    degrees=True,
                ).as_matrix()
                shifts = array_api_compat.to_device(shifts, "cpu")
                shifts = np.asarray(shifts)
                associated_translations = -(np.linalg.inv(rot) @ shifts[:, :, None])
                associated_translations = associated_translations[:, :, 0].reshape(
                    len(indices_axes), len(indices_rot), 3
                )

                # Go back to CPU
                def to_numpy(x):
                    return np.asarray(array_api_compat.to_device(x, "cpu"))

                energies = energies.reshape(len(indices_axes), len(indices_rot))
                psf_inverse_transformed_fft = psf_inverse_transformed_fft.reshape(
                    len(indices_axes), len(indices_rot), *image_shape
                )
                view_inverse_transformed_shifted_fft = (
                    view_inverse_transformed_shifted_fft.reshape(
                        len(indices_axes), len(indices_rot), *image_shape
                    )
                )
                (
                    energies,
                    psf_inverse_transformed_fft,
                    view_inverse_transformed_shifted_fft,
                ) = map(
                    to_numpy,
                    (
                        energies,
                        psf_inverse_transformed_fft,
                        view_inverse_transformed_shifted_fft,
                    ),
                )

                # Find the energy minimum
                j, k = np.unravel_index(np.argmin(energies), energies.shape)
                min_energy = energies[j, k]
                energies_batch.append(min_energy)  # store it
                energies_each_view[v].append(min_energy)
                pbar2.set_description(f"particle {v} energy : {min_energy:.1f}")

                # The transformation associated with that minimum
                best_idx_axes, best_idx_rot = indices_axes[j], indices_rot[k]
                rot_vec = [
                    thetas[best_idx_axes],
                    phis[best_idx_axes],
                    psis[best_idx_rot],
                ]
                estimated_poses_iter[v, :3] = rot_vec  # store the rotation
                estimated_poses_iter[v, 3:] = associated_translations[
                    j, k
                ]  # store the translation

                # Compute the gradient of the energy with respect to the volume
                grad = compute_grad(
                    volume_representation.volume_fourier,
                    psf_inverse_transformed_fft[j, k],
                    view_inverse_transformed_shifted_fft[j, k],
                )
                gradients_batch.append(grad)  # accumulate

                # Update the importance distributions
                energies = normalize(energies, max=6)
                likelihoods = np.exp(-energies)
                phi_axes = likelihoods.dot(1 / imp_distrs_rot[v][indices_rot]) / N_rot
                phi_rot = (
                    likelihoods.T.dot(1 / imp_distrs_axes[v][indices_axes]) / N_axes
                )
                K_axes = np.exp(
                    params_learning_alg.coeff_kernel_axes
                    * axes[:, indices_axes].T.dot(axes)
                )
                K_rot = np.zeros((N_rot, M_rot))
                for k, idx_rot in enumerate(indices_rot):
                    a = psis[idx_rot]
                    if params_learning_alg.gaussian_kernel:
                        K_rot[k, :] = one_d_gaussian(
                            psis, a, params_learning_alg.coeff_kernel_rot
                        )
                    else:
                        K_rot[k, :] = np.exp(
                            np.cos(a - psis) * params_learning_alg.coeff_kernel_rot
                        )
                update_imp_distr(
                    imp_distrs_axes, phi_axes, K_axes, unif_prop_axes, M_axes, v
                )
                update_imp_distr(
                    imp_distrs_rot, phi_rot, K_rot, unif_prop_rot, M_rot, v
                )

            # Weighted gradient descent
            energies_batch = np.array(energies_batch)
            m = energies_batch.max()
            e = np.exp(-params_learning_alg.beta_grad * (energies_batch - m))
            grad_weights = e / e.sum()
            grad = (grad_weights * np.stack(gradients_batch, axis=-1)).sum(axis=-1)
            volume_representation.gd_step(
                grad, params_learning_alg.lr, params_learning_alg.reg_coeff
            )

            # Center of mass centered
            shift = volume_representation.center()
            for v in range(estimated_poses_iter.shape[0]):
                estimated_poses_iter[v, 3:] -= shift

            # Increase energy
            total_energy += np.sum(energies_batch)

        ests_poses.append(estimated_poses_iter)
        pbar2.close()

        if (
            params_learning_alg.epochs_of_suppression is not None
            and len(params_learning_alg.epochs_of_suppression) > 0
            and itr == params_learning_alg.epochs_of_suppression[0]
        ):
            nb_step_of_supress += 1
            prop_to_suppress = params_learning_alg.proportion_of_views_suppressed.pop(0)
            nb_views_to_suppress = int(len(views) * prop_to_suppress)
            params_learning_alg.epochs_of_suppression.pop(0)
            energies_each_views_current_iter = np.array(energies_each_view)[:, -1]
            # print('energies each views', energies_each_views_current_iter)
            idx_views_to_keep = np.argsort(energies_each_views_current_iter)[
                : len(energies_each_views_current_iter) - nb_views_to_suppress
            ]
            # print('idx kepts', idx_views_to_keep)
            views = [views[idx] for idx in idx_views_to_keep]
            imp_distrs_axes = [imp_distrs_axes[idx] for idx in idx_views_to_keep]
            imp_distrs_rot = [imp_distrs_rot[idx] for idx in idx_views_to_keep]
            energies_each_view = [energies_each_view[idx] for idx in idx_views_to_keep]
            recorded_shifts = [recorded_shifts[idx] for idx in idx_views_to_keep]
            file_names = [file_names[idx] for idx in idx_views_to_keep]
            folder_views_selected_step = (
                f"{folder_views_selected}/step_{nb_step_of_supress}"
            )
            make_dir(folder_views_selected_step)
            for i, fn in enumerate(file_names):
                save(f"{folder_views_selected_step}/{fn}", views[i])

        # Register reconstrution with groundtruth and save it
        volume_representation.register_and_save(
            sub_dir,
            f"recons_epoch_{itr}.tif",
            ground_truth=ground_truth,
            one_component=False,
            gpu=gpu,
        )

        # Update uniform distribution
        unif_prop_axes /= params_learning_alg.dec_prop
        unif_prop_rot /= params_learning_alg.dec_prop
        if params_learning_alg.N_iter_with_unif_distr is not None:
            if itr > params_learning_alg.N_iter_with_unif_distr:
                unif_prop_axes, unif_prop_rot = 0, 0
        if unif_prop_axes < unif_prop_min:
            unif_prop_axes = unif_prop_min
        if unif_prop_rot < unif_prop_min:
            unif_prop_rot = unif_prop_min

        # Save stuff
        write_array_csv(
            estimated_poses_iter, f"{sub_dir}/estimated_poses_epoch_{itr}.csv"
        )
        if ground_truth is not None:
            regist_im = io.imread(os.path.join(sub_dir, f"recons_epoch_{itr}.tif"))
            ssim_gt_recons = ssim(normalize(ground_truth), normalize(regist_im))
            ssims.append(ssim_gt_recons)
        imp_distrs_rot_recorded.append(copy.deepcopy(imp_distrs_rot))
        imp_distrs_axes_recorded.append(copy.deepcopy(imp_distrs_axes))
        total_energy /= epoch_length
        recorded_energies.append(total_energy)

        pbar.set_description(f"energy : {total_energy:.1f}")
        pbar.update()

    if itr > 0:
        shutil.copyfile(
            os.path.join(sub_dir, f"recons_epoch_{itr}.tif"),
            os.path.join(output_dir, "final_recons.tif"),
        )
        shutil.copyfile(
            os.path.join(sub_dir, f"estimated_poses_epoch_{itr}.csv"),
            os.path.join(output_dir, "poses.csv"),
        )
    pbar.close()
    write_array_csv(np.array(ssims), f"{output_dir}/ssims.csv")

    energies_each_view = np.array(energies_each_view)
    np.save(os.path.join(output_dir, "energies_each_view.npy"), energies_each_view)
    if len(imp_distrs_rot_recorded) > 0:
        imp_distrs_rot_recorded = np.stack(imp_distrs_rot_recorded, axis=0)
    if len(imp_distrs_axes_recorded) > 0:
        imp_distrs_axes_recorded = np.stack(imp_distrs_axes_recorded, axis=0)
    np.save(
        os.path.join(output_dir, "distributions_rot.npy"),
        imp_distrs_rot_recorded,
    )
    np.save(
        os.path.join(output_dir, "distributions_axes.npy"),
        imp_distrs_axes_recorded,
    )
    data2 = pd.DataFrame({"energy": recorded_energies})
    data2.to_csv(os.path.join(output_dir, "energies.csv"))
    params_to_save = params_learning_alg.__dict__.copy()
    del params_to_save["params"]
    del params_to_save["dtype"]
    with open(os.path.join(output_dir, "params_learning_alg.json"), "w") as f:
        json.dump(params_to_save, f)

    return (
        imp_distrs_rot_recorded,
        imp_distrs_axes_recorded,
        recorded_energies,
        recorded_shifts,
        unif_prop,
        volume_representation,
        itr,
        energies_each_view,
        views,
        file_names,
        estimated_poses_iter,
    )


def update_imp_distr(imp_distr, phi, K, prop, M, v):
    # phi = phi ** (1 / temp)
    q_first_comp = phi @ K
    q_first_comp /= np.sum(q_first_comp)
    imp_distr[v] = (1 - prop) * q_first_comp + prop * np.ones(M) / M
    return q_first_comp


def compute_shifts(
    reference_volume_fft: Array,
    psf: Array,
    inverse_transforms: Array,
    view: Array,
    interp_order: int = 1,
) -> Tuple[Array, Array, Array]:
    xp = array_api_compat.array_namespace(inverse_transforms, view)
    if xp == array_api_compat.torch:

        def pytorch_fftn_wrapper(x, s=None, axes=None, norm="backward"):
            from torch.fft import fftn as fftn_torch

            return fftn_torch(x, s=s, dim=axes, norm=norm)

        fftn = pytorch_fftn_wrapper
    else:
        fftn = xp.fft.fftn

    (device,) = set([str(xp.device(inverse_transforms)), str(xp.device(view))])
    psf = xp.asarray(psf, device=device)
    volume_fourier = xp.asarray(reference_volume_fft, device=device)

    N = inverse_transforms.shape[0]
    image_shape = psf.shape

    # Rotate the psf backward
    psfs_rotated = rotation(
        xp.broadcast_to(psf, (N,) + image_shape),
        inverse_transforms,
        order=interp_order,
    )
    psfs_rotated_fft = fftn(psfs_rotated, axes=(1, 2, 3))

    # Rotate the view back to the volume
    view_rotated = rotation(
        xp.broadcast_to(view, (N,) + image_shape),
        inverse_transforms,
        order=interp_order,
    )
    view_rotated_fft = fftn(view_rotated, axes=(1, 2, 3))

    shift, _, _ = phase_cross_correlation(
        psfs_rotated_fft * volume_fourier,
        view_rotated_fft,
        nb_spatial_dims=3,
        upsample_factor=10,
        normalization=None,
        space="fourier",
    )
    shift = xp.stack(shift, axis=-1)
    return shift, psfs_rotated_fft, view_rotated_fft


def compute_energy(reference_volume_fft, psf_rotated_fft, view_rotated_fft):
    xp = array_api_compat.array_namespace(view_rotated_fft)
    device = xp.device(view_rotated_fft)
    reference_volume_fft = xp.asarray(reference_volume_fft, device=device)
    psf_rotated_fft = xp.asarray(psf_rotated_fft, device=device)
    image_size = xp.prod(xp.asarray(reference_volume_fft.shape[-3:], device=device))
    energy = (
        xp.linalg.vector_norm(
            psf_rotated_fft * reference_volume_fft - view_rotated_fft,
            axis=(-3, -2, -1),
        )
        ** 2
        / image_size
    )
    return energy


def compute_grad(volume_fourier, psf_rotated_fft, view_rotated_fft):
    grad = psf_rotated_fft * (psf_rotated_fft * volume_fourier - view_rotated_fft)
    return grad
