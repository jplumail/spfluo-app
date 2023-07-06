import copy
import json
import os
import shutil

import cupy as cp
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R
from skimage import io
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm

from ..common_image_processing_methods.others import normalize, stopping_criteria
from ..common_image_processing_methods.rotation_translation import (
    conversion_2_first_eulers_angles_cartesian,
    get_rotation_matrix,
)
from ..manage_files.read_save_files import make_dir, save, write_array_csv
from ..volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import (
    one_d_gaussian,
)


def gradient_descent_importance_sampling_known_axes(
    volume_representation,
    uniform_sphere_discretization,
    true_rot_vecs,
    true_trans_vecs,
    views,
    imp_distrs_rot,
    unif_prop,
    unif_prop_min,
    params_learning_alg,
    known_trans,
):
    thetas, phis, psis = uniform_sphere_discretization
    M_rot = len(psis)
    imp_distrs_rot_recorded = []
    itr = 0
    recorded_energies = []
    nb_views = len(views)
    recorded_shifts = [[] for _ in range(nb_views)]
    views_fft = [np.fft.fftn(v) for v in views]
    while itr < params_learning_alg.N_iter_max and (
        not stopping_criteria(recorded_energies, params_learning_alg.eps)
    ):
        itr += 1
        total_energy = 0
        for v in range(nb_views):
            indices_rot = np.random.choice(
                range(M_rot), p=imp_distrs_rot[v], size=params_learning_alg.N_rot
            )
            energies = np.zeros(params_learning_alg.N_rot)
            best_energy = 10**10
            best_idx_rot = 0
            for k, idx_rot in enumerate(indices_rot):
                psi = psis[idx_rot]
                rot_vec = [true_rot_vecs[v, 0], true_rot_vecs[v, 1], psi]
                rot_mat = get_rotation_matrix(rot_vec, params_learning_alg.convention)
                # energy = volume_representation.\
                # get_energy_2(rot_mat, true_trans_vecs[v], views_fft[v], known_trans)
                energy, _ = volume_representation.get_energy(
                    rot_mat,
                    true_trans_vecs[v],
                    views[v],
                    recorded_shifts,
                    v,
                    False,
                    known_trans,
                )

                energies[k] = energy
                if energy < best_energy:
                    best_energy = energy
                    best_idx_rot = idx_rot
            psi = psis[best_idx_rot]
            rot_vec = [true_rot_vecs[v, 0], true_rot_vecs[v, 1], psi]
            rot_mat = get_rotation_matrix(rot_vec, params_learning_alg.convention)
            energy = volume_representation.one_gd_step(
                rot_mat,
                true_trans_vecs[v],
                views,
                params_learning_alg.lr,
                known_trans,
                v,
                recorded_shifts,
            )  # ,
            # suppress_gauss=(v==nb_views-1))

            total_energy += energy
            energies = normalize(energies, max=6)
            likekihoods = np.exp(-energies)
            K = np.zeros((params_learning_alg.N_rot, M_rot))
            for k, idx_rot in enumerate(indices_rot):
                a = psis[idx_rot]
                K[k, :] = one_d_gaussian(psis, a, params_learning_alg.std_rot)
            update_imp_distr(imp_distrs_rot, likekihoods, K, unif_prop, M_rot, v)
        imp_distrs_rot_recorded.append(copy.deepcopy(imp_distrs_rot))
        unif_prop /= params_learning_alg.dec_factor
        if unif_prop < unif_prop_min:
            unif_prop = unif_prop_min
        total_energy /= nb_views
        # print(f"total energy epoque {itr}", total_energy)
        recorded_energies.append(total_energy)
    return (
        imp_distrs_rot_recorded,
        recorded_energies,
        recorded_shifts,
        unif_prop,
        volume_representation,
        itr,
    )


def gradient_descent_importance_sampling_known_rot(
    volume_representation,
    uniform_sphere_discretization,
    true_rot_vecs,
    true_trans_vecs,
    views,
    imp_distrs,
    unif_prop,
    unif_prop_min,
    params_learning_alg,
    known_trans,
):
    thetas, phis, psis = uniform_sphere_discretization
    x, y, z = conversion_2_first_eulers_angles_cartesian(thetas, phis)
    axes = np.array([x, y, z])
    M = len(thetas)
    imp_distrs_recorded = []
    itr = 0
    recorded_energies = []
    nb_views = len(views)
    recorded_shifts = [[] for _ in range(nb_views)]
    while itr < params_learning_alg.N_iter_max and (
        not stopping_criteria(recorded_energies, params_learning_alg.eps)
    ):
        itr += 1
        total_energy = 0
        for v in range(nb_views):
            indices = np.random.choice(
                range(M), p=imp_distrs[v], size=params_learning_alg.N_axes
            )
            energies = np.zeros(params_learning_alg.N_axes)
            best_energy = 10**10
            best_idx = 0
            for k, idx in enumerate(indices):
                theta, phi = thetas[idx], phis[idx]
                rot_vec = [theta, phi, true_rot_vecs[v, 2]]
                rot_mat = get_rotation_matrix(rot_vec, params_learning_alg.convention)
                energy, _ = volume_representation.get_energy(
                    rot_mat,
                    true_trans_vecs[v],
                    views[v],
                    recorded_shifts,
                    v,
                    False,
                    known_trans,
                )

                energies[k] = energy
                if energy < best_energy:
                    best_energy = energy
                    best_idx = idx

            theta, phi = thetas[best_idx], phis[best_idx]
            rot_vec = [theta, phi, true_rot_vecs[v, 2]]
            rot_mat = get_rotation_matrix(rot_vec, params_learning_alg.convention)
            energy = volume_representation.one_gd_step(
                rot_mat,
                true_trans_vecs[v],
                views,
                params_learning_alg.lr,
                known_trans,
                v,
                recorded_shifts,
            )
            total_energy += energy
            energies = normalize(energies, max=6)
            likekihoods = np.exp(-energies)
            K_axes = np.exp(
                params_learning_alg.coeff_kernel * axes[:, indices].T.dot(axes)
            )

            update_imp_distr(imp_distrs, likekihoods, K_axes, unif_prop, M, v)
        imp_distrs_recorded.append(copy.deepcopy(imp_distrs))
        unif_prop /= params_learning_alg.dec_factor
        if unif_prop < unif_prop_min:
            unif_prop = unif_prop_min
        total_energy /= nb_views
        # print(f'total energy epoque {itr}', total_energy)
        recorded_energies.append(total_energy)
    return (
        imp_distrs_recorded,
        recorded_energies,
        recorded_shifts,
        unif_prop,
        volume_representation,
        itr,
    )


def gd_importance_sampling_3d(
    volume_representation,
    uniform_sphere_discretization,
    true_trans_vecs,
    views,
    imp_distrs_axes,
    imp_distrs_rot,
    unif_prop,
    unif_prop_min,
    params_learning_alg,
    known_trans,
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
    thetas, phis, psis = uniform_sphere_discretization
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
    ests_rot_vecs = []
    nb_step_of_supress = 0
    pbar = tqdm(total=params_learning_alg.N_iter_max, leave=False, desc="energy : +inf")
    views = views.astype(params_learning_alg.dtype)
    while itr < params_learning_alg.N_iter_max and (
        not stopping_criteria(recorded_energies, params_learning_alg.eps)
    ):
        # print(f'nb views epoch {itr} : ', len(views))

        nb_views = len(views)
        itr += 1
        total_energy = 0
        estimated_rot_vecs_iter = np.zeros((nb_views, 3))

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
                indices_axes = np.random.choice(
                    range(M_axes), p=imp_distrs_axes[v], size=params_learning_alg.N_axes
                )
                indices_rot = np.random.choice(
                    range(M_rot), p=imp_distrs_rot[v], size=params_learning_alg.N_rot
                )
                energies = np.zeros(
                    (params_learning_alg.N_axes, params_learning_alg.N_rot)
                )
                if known_trans:
                    true_trans_vec = true_trans_vecs[v]
                else:
                    true_trans_vec = None

                if gpu == "pytorch" or gpu == "cucim":
                    rot_vecs = np.stack(
                        np.broadcast_arrays(
                            thetas[indices_axes][:, None],
                            phis[indices_axes][:, None],
                            psis[indices_rot],
                        ),
                        axis=-1,
                    )  # Na x Nr x 3
                    rot_mats = R.from_euler(
                        "zxz", rot_vecs.reshape(-1, 3), degrees=True
                    ).as_matrix()  # (Na*Nr) x 3 x 3
                if gpu == "pytorch":
                    rot_mats_gpu, true_trans_vec_gpu, view_gpu = map(
                        lambda x: torch.as_tensor(
                            x.astype(params_learning_alg.dtype), device="cuda"
                        )
                        if x is not None
                        else None,
                        [rot_mats, true_trans_vec, views[v]],
                    )
                    energies, _ = volume_representation.get_energy_gpu_pytorch(
                        rot_mats_gpu,
                        true_trans_vec_gpu,
                        view_gpu,
                        None,
                        None,
                        False,
                        known_trans,
                        interp_order=params_learning_alg.interp_order,
                    )
                    energies = (
                        energies.cpu()
                        .numpy()
                        .reshape(len(indices_axes), len(indices_rot))
                    )
                elif gpu == "cucim":
                    rot_mats_gpu, true_trans_vec_gpu, view_gpu = map(
                        lambda x: cp.array(x.astype(params_learning_alg.dtype))
                        if x is not None
                        else None,
                        [rot_mats, true_trans_vec, views[v]],
                    )
                    rot_mats_gpu = rot_mats_gpu.reshape(
                        params_learning_alg.N_axes, params_learning_alg.N_rot, 3, 3
                    )
                    energies = cp.array(energies)
                    for j, idx_axes in enumerate(indices_axes):
                        for k, idx_rot in enumerate(indices_rot):
                            energies[j, k], _ = volume_representation.get_energy_gpu(
                                rot_mats_gpu[j, k],
                                true_trans_vec,
                                view_gpu,
                                None,
                                None,
                                False,
                                known_trans,
                                interp_order=params_learning_alg.interp_order,
                            )
                    energies = energies.get()
                else:
                    for j, idx_axes in enumerate(indices_axes):
                        for k, idx_rot in enumerate(indices_rot):
                            rot_vec = [thetas[idx_axes], phis[idx_axes], psis[idx_rot]]
                            rot_mat = get_rotation_matrix(
                                rot_vec, params_learning_alg.convention
                            ).astype(params_learning_alg.dtype)
                            energy, _ = volume_representation.get_energy(
                                rot_mat,
                                true_trans_vec,
                                views[v],
                                recorded_shifts,
                                v,
                                False,
                                known_trans,
                                interp_order=params_learning_alg.interp_order,
                            )
                            energies[j, k] = energy

                j, k = np.unravel_index(np.argmin(energies), energies.shape)
                best_idx_axes, best_idx_rot = indices_axes[j], indices_rot[k]
                energies = normalize(energies, max=6)
                likelihoods = np.exp(-energies)
                rot_vec = [
                    thetas[best_idx_axes],
                    phis[best_idx_axes],
                    psis[best_idx_rot],
                ]

                estimated_rot_vecs_iter[v, :] = rot_vec
                rot_mat = get_rotation_matrix(rot_vec, params_learning_alg.convention)
                grad, energy = volume_representation.compute_grad(
                    rot_mat,
                    true_trans_vec,
                    views,
                    known_trans,
                    v,
                    interp_order=params_learning_alg.interp_order,
                )
                energies_batch.append(energy)
                gradients_batch.append(grad)
                pbar2.set_description(f"particle {v} energy : {energy:.1f}")
                energies_each_view[v].append(energy)
                phi_axes = (
                    likelihoods.dot(1 / imp_distrs_rot[v][indices_rot])
                    / params_learning_alg.N_rot
                )
                phi_rot = (
                    likelihoods.T.dot(1 / imp_distrs_axes[v][indices_axes])
                    / params_learning_alg.N_axes
                )
                K_axes = np.exp(
                    params_learning_alg.coeff_kernel_axes
                    * axes[:, indices_axes].T.dot(axes)
                )
                K_rot = np.zeros((params_learning_alg.N_rot, M_rot))

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

            # Gradient descent
            energies_batch = np.array(energies_batch)
            m = energies_batch.max()
            e = np.exp(-params_learning_alg.beta_grad * (energies_batch - m))
            grad_weights = e / e.sum()
            grad = (grad_weights * np.stack(gradients_batch, axis=-1)).sum(axis=-1)
            volume_representation.gd_step(
                grad, params_learning_alg.lr, params_learning_alg.reg_coeff
            )

            # Increase energy
            total_energy += np.sum(energies_batch)

        ests_rot_vecs.append(estimated_rot_vecs_iter)
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
            translate=True,
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
            estimated_rot_vecs_iter, f"{sub_dir}/estimated_rot_vecs_epoch_{itr}.csv"
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

    shutil.copyfile(
        os.path.join(sub_dir, f"recons_epoch_{itr}.tif"),
        os.path.join(sub_dir, f"final_recons.tif"),
    )
    pbar.close()
    write_array_csv(np.array(ssims), f"{output_dir}/ssims.csv")

    np.save(
        os.path.join(output_dir, "energies_each_view.npy"), np.array(energies_each_view)
    )
    np.save(
        os.path.join(output_dir, "distributions_rot.npy"),
        np.stack(imp_distrs_rot_recorded, axis=0),
    )
    np.save(
        os.path.join(output_dir, "distributions_axes.npy"),
        np.stack(imp_distrs_axes_recorded, axis=0),
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
        ests_rot_vecs,
    )


def update_imp_distr(imp_distr, phi, K, prop, M, v):
    # phi = phi ** (1 / temp)
    q_first_comp = phi @ K
    q_first_comp /= np.sum(q_first_comp)
    imp_distr[v] = (1 - prop) * q_first_comp + prop * np.ones(M) / M
    return q_first_comp
