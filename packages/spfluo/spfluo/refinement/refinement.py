"""Some functions from this file were translated from code written by Denis Fortun"""

import itertools
import logging
import math
import time
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import spfluo.utils.debug as debug
from spfluo.utils.array import array_namespace, get_device, to_device, to_numpy
from spfluo.utils.memory import split_batch
from spfluo.utils.transform import (
    get_transform_matrix,
    get_transform_matrix_from_pose,
    symmetrize_poses,
)
from spfluo.utils.volume import (
    affine_transform,
    discretize_sphere_uniformly,
    interpolate_to_size,
    pad,
    phase_cross_correlation,
)

if TYPE_CHECKING:
    from spfluo.utils.array import Array, Device, array_api_module

refinement_logger = logging.getLogger("spfluo.refinement")


def rotate(
    volumes: "Array", poses: "Array", inverse=False, batch=True, multichannel=False
):
    xp = array_namespace(volumes, poses)
    H = get_transform_matrix(
        volumes.shape[-3:],
        poses[..., :3],
        poses[..., 3:],
        convention="XZX",
        degrees=True,
    )
    if not inverse:  # scipy's affine_transform do inverse transform by default
        H = xp.linalg.inv(H)
    H = xp.asarray(H, dtype=volumes.dtype)
    return affine_transform(
        volumes, H, order=1, prefilter=False, batch=batch, multichannel=multichannel
    )


def reconstruction_L2(
    volumes: "Array",
    psf: "Array",
    poses: "Array",
    lambda_: "Array",
    batch: bool = False,
    symmetry: bool = False,
    device: Optional["Device"] = None,
    batch_size: Optional[int] = None,
    multichannel: bool = False,
):
    """Reconstruct a particule from volumes and their poses.
    M reconstructions can be done at once.

    Args:
        volumes (Array): stack of N 3D images of shape (N, (C), D, D, D)
        psf (Array): 3D image of shape ((C), d, h, w)
        poses (Array):
            stack(s) of N poses of shape ((M), (k), N, 6)
            A 'pose' is represented by 6 numbers
                euler angles in the 'zxz' convention in degrees
                a translation vector tz, ty, tx
                you need N poses to describe your volumes
            k is the degree of symmetry. Optional
                If your volumes have a ninefold symmetry, you might want to multiply
                the number of poses by 9 to get a better reconstruction. For that,
                you will need to symmetry axis. It will also increase compute time.
            M is the number of reconstructions you want to do. Optional
                Usefull for computing several reconstructions from the same
                set of volumes
        lambda (Array): regularization parameters of shape ((M),)
        batch (bool): do M reconstructions at once
            poses must be of shape (M, (k), N, 6), the k dim is optional
        symmetry (bool): use a k-degree symmetry
            poses must be of shape ((M), k, N, 6), the M dim is optional
        device (Device): the device to do the computation on.
        batch_size (int or None) : if None, do all the computation at once
        multichannel (bool): choose if multichannel volumes and psf.
            if True, volumes must be of shape (N, C, D, D, D)

    Returns:
        recon (Array):
            reconstruction(s) of shape ((M), (C), D, D, D) or ((M), (C), D+1, D+1, D+1)
    """
    refinement_logger.info("Calling function reconstruction_L2")

    xp = array_namespace(volumes, poses, psf, lambda_)
    host_device = get_device(volumes)
    compute_device = device

    if multichannel:
        N, C, D, H, W = volumes.shape
        c, d, h, w = psf.shape
        assert c == C
    else:
        N, D, H, W = volumes.shape
        d, h, w = psf.shape
        volumes = volumes[:, None, ...]
        psf = psf[None, ...]
        C = 1
    assert D == H == W

    floating_dtype = volumes.dtype
    assert xp.isdtype(floating_dtype, "real floating")
    assert floating_dtype == psf.dtype
    if floating_dtype == xp.float32:
        complex_dtype = xp.complex64
    elif floating_dtype == xp.float64:
        complex_dtype = xp.complex128
    else:
        raise ValueError(
            f"{floating_dtype=} is not supported. "
            "Only float32 and float64 are accepted."
        )

    refinement_logger.debug(f"floating dtype: {floating_dtype}")
    refinement_logger.debug(f"complex dtype: {complex_dtype}")

    refinement_logger.info(
        "Arguments:"
        f" {N} volumes of size {D}x{D}x{D} with C channels"
        f" PSF of size {d}x{h}x{w}"
    )
    if batch:
        assert poses.ndim > 2
        M = poses.shape[0]
        assert lambda_.shape[0] == M
        refinement_logger.info(
            f"Running in batch mode: {M} reconstructions will be done"
        )
    else:
        M = 1
        poses = xp.expand_dims(poses, axis=0)
        lambda_ = xp.reshape(lambda_, (1,))

    if symmetry:
        assert poses.ndim == 4
        k = poses.shape[1]
        refinement_logger.info(f"Symmetry enabled, of degree k={k}")
    else:
        k = 1
        poses = xp.expand_dims(poses, axis=1)  # shape (1, 1, N, 6)

    new_poses = xp.asarray(poses, copy=True)
    if D % 2 == 0:
        # pad by 1 pixel on the right
        volumes = pad(volumes, ((0, 0), (0, 0), (0, 1), (0, 1), (0, 1)))
        assert volumes.shape == (N, C, D + 1, D + 1, D + 1)
        D = D + 1
        new_poses[..., 3:] -= 0.5
        refinement_logger.info(f"Reshaped volumes to odd size {D}x{D}x{D}")
        resize = True
    else:
        resize = False

    psf = interpolate_to_size(psf, (D, D, D), multichannel=True)

    num = xp.zeros((M, C, D, D, D), dtype=complex_dtype, device=host_device)
    den = xp.zeros_like(num, dtype=floating_dtype)

    poses_psf = xp.zeros_like(new_poses)
    poses_psf[..., :3] = new_poses[..., :3]

    # Move data to compute device
    psf = to_device(psf, compute_device)
    for (start1, end1), (start2, end2), (start3, end3), (start4, end4) in split_batch(
        (M, k, N, C), batch_size
    ):
        number_poses = (end1 - start1) * (end2 - start2)
        number_volumes = end3 - start3
        number_channels = end4 - start4

        # Move data to compute device
        poses_minibatch = to_device(
            new_poses[start1:end1, start2:end2, start3:end3, ...], compute_device
        )
        poses_psf_minibatch = to_device(
            poses_psf[start1:end1, start2:end2, start3:end3, ...], compute_device
        )
        volumes_minibatch = to_device(
            volumes[start3:end3, start4:end4, ...], compute_device
        )
        psf_minibatch = to_device(psf[start4:end4, ...], compute_device)

        y = xp.stack(
            (volumes_minibatch,) * number_poses,
        )
        y = xp.reshape(
            rotate(
                xp.reshape(
                    y, (number_poses * number_volumes, number_channels, D, D, D)
                ),
                xp.reshape(poses_minibatch, (number_poses * number_volumes, 6)),
                inverse=True,
                multichannel=True,
            ),
            (end1 - start1, end2 - start2, number_volumes, number_channels, D, D, D),
        )

        H_ = xp.stack((psf_minibatch,) * number_volumes * number_poses)
        H_ = xp.reshape(
            rotate(
                H_,
                xp.reshape(poses_psf_minibatch, (number_poses * number_volumes, 6)),
                inverse=True,
                multichannel=True,
            ),
            (end1 - start1, end2 - start2, end3 - start3, number_channels, D, D, D),
        )

        H_ = xp.fft.fftn(xp.astype(H_, complex_dtype), axes=(-3, -2, -1))

        # Compute numerator
        y = xp.astype(y, complex_dtype)
        y = xp.fft.fftn(xp.fft.fftshift(y, axes=(-3, -2, -1)), axes=(-3, -2, -1))
        y = xp.conj(H_) * y
        num[start1:end1, start4:end4, ...] += to_device(
            xp.sum(y, axis=(1, 2), dtype=complex_dtype), host_device
        )  # reduce symmetry and N dims

        # Compute denominator
        den[start1:end1, start4:end4, ...] += to_device(
            xp.sum(xp.abs(H_) ** 2, axis=(1, 2), dtype=floating_dtype), host_device
        )
        del H_

    num = num / (N * k)
    den = den / (N * k)

    dxyz = xp.zeros((3, 2, 2, 2), device=host_device, dtype=complex_dtype)
    dxyz[0, 0, 0, 0] = 1
    dxyz[0, 1, 0, 0] = -1
    dxyz[1, 0, 0, 0] = 1
    dxyz[1, 0, 1, 0] = -1
    dxyz[2, 0, 0, 0] = 1
    dxyz[2, 0, 0, 1] = -1

    dxyz_padded = pad(dxyz, ((0, 0), *(((D - 1) // 2, (D - 2) // 2),) * 3))
    D_ = xp.fft.fftn(dxyz_padded, axes=(1, 2, 3))
    DtD = xp.sum(
        (xp.abs(D_) ** 2),
        axis=0,
        dtype=floating_dtype,
    )

    den += xp.asarray(lambda_[:, None, None, None, None] * DtD, dtype=xp.float32)

    num = xp.fft.ifftn(num / den, axes=(-3, -2, -1))
    refinement_logger.debug(f"num dtype: {num.dtype}")
    recon = xp.real(num)
    refinement_logger.debug(f"recon 1 dtype: {recon.dtype}")
    recon = xp.where(
        recon > 0, recon, xp.asarray(0.0, device=host_device, dtype=floating_dtype)
    )
    refinement_logger.debug(f"recon 2 end dtype: {recon.dtype}")

    if resize:
        recon = interpolate_to_size(
            recon, (D - 1, D - 1, D - 1), batch=True, multichannel=True
        )
    refinement_logger.debug(f"recon 3 end dtype: {recon.dtype}")

    if not multichannel:
        recon = recon[:, 0, ...]
    if not batch:
        recon = recon[0, ...]

    if refinement_logger.isEnabledFor(logging.DEBUG):
        p = debug.save_image(
            to_numpy(recon),
            debug.DEBUG_DIR_REFINEMENT,
            reconstruction_L2,
            "reconstruction",
            sequence=batch,
            multichannel=True,
        )
        refinement_logger.debug("Saving reconstruction(s) at " + str(p))

    refinement_logger.debug(f"recon end dtype: {recon.dtype}")
    return recon


def convolution_matching_poses(
    reference: "Array",
    volumes: "Array",
    psf: "Array",
    potential_poses: "Array",
    device: Optional["Device"] = None,
    batch_size: int = 1,
):
    """Find the best pose from a list of poses for each volume.
    There can be a different list of pose for each volume.

    Args:
        reference (Array): reference 3D image of shape (C, D, D, D)
        volumes (Array): volumes to match of shape (N, C, D, D, D)
        psf (Array): 3D PSF of shape (C, d, h, w)
        potential_poses (Array): poses to test of shape (N, M, 6)
        device (Device): the device to do the computation on.
        batch_size (int): the batch size
    
    Returns:
        A tuple of 2 arrays representing the best poses for each volume 
        of shape (N, 6) and the dftRegistration error associated
        to each pose (N,).
    """
    xp = array_namespace(volumes, psf, potential_poses)
    host_device = get_device(volumes)
    compute_device = device

    # Shapes
    N1, M, x = potential_poses.shape
    assert x == 6
    N, C, D, H, W = volumes.shape
    assert D == H == W
    assert N == N1
    C1, D1, H, W = reference.shape
    assert C1 == C and D1 == D and H == D and W == D
    C2, _, _, _ = psf.shape
    assert C2 == C

    new_potential_poses = xp.asarray(potential_poses, copy=True)
    if D % 2 == 0:
        new_potential_poses[..., 3:] += 0.5
    else:
        new_potential_poses[..., 3:] += 1.0

    # PSF
    h = xp.fft.fftn(
        xp.fft.fftshift(
            interpolate_to_size(psf, (D, D, D), multichannel=True), axes=(1, 2, 3)
        ),
        axes=(1, 2, 3),
    )

    shifts = xp.empty((N, M, 3), dtype=reference.dtype, device=host_device)
    errors = xp.empty((N, M), dtype=reference.dtype, device=host_device)

    # Move data to compute device
    h = to_device(h, compute_device)
    reference = to_device(reference, compute_device)
    for (start1, end1), (start2, end2) in tqdm(split_batch((N, M), batch_size), total=(N*M)//batch_size, leave=False, desc="convolution matching"):
        minibatch_size = (end1 - start1) * (end2 - start2)
        potential_poses_minibatch = to_device(
            new_potential_poses[start1:end1, start2:end2], compute_device
        )

        # Volumes to Fourier space
        volumes_freq = xp.fft.fftn(
            to_device(volumes[start1:end1], compute_device), axes=(2, 3, 4)
        )

        # Rotate the reference
        reference_minibatch = xp.stack((reference,) * minibatch_size)
        reference_minibatch = xp.reshape(
            rotate(
                reference_minibatch,
                xp.reshape(potential_poses_minibatch, (minibatch_size, 6)),
                multichannel=True,
            ),
            (end1 - start1, end2 - start2, C, D, D, D),
        )
        reference_minibatch = h * xp.fft.fftn(reference_minibatch, axes=(3, 4, 5))

        # Registration
        sh, err, _ = phase_cross_correlation(
            reference_minibatch,
            volumes_freq[:, None],
            nb_spatial_dims=3,
            normalization=None,
            upsample_factor=10,
            space="fourier",
            multichannel=True,
        )
        sh = xp.stack(list(sh), axis=-1)

        errors[start1:end1, start2:end2] = to_device(err, host_device)
        shifts[start1:end1, start2:end2] = to_device(sh, host_device)

        del volumes_freq, reference_minibatch, err, sh

    errors, best_indices = xp.min(errors, axis=1), xp.argmin(errors, axis=1)
    best_poses = potential_poses[xp.arange(N), best_indices]
    best_poses[:, 3:] = -shifts[xp.arange(N), best_indices]
    return best_poses, errors


def find_L(precision: float):
    return math.ceil(((360 / precision) ** 2) / math.pi)


def create_poses_grid(
    xp: "array_api_module", M_axes: int, M_rot: int, symmetry: int = 1, **array_kwargs
):
    (theta, phi, psi), precision = discretize_sphere_uniformly(
        xp, M_axes, M_rot, product=True, symmetry=symmetry, **array_kwargs
    )
    list_angles = xp.stack([theta, phi, psi], axis=-1)
    M = list_angles.shape[0]
    list_translation = xp.zeros((M, 3), **array_kwargs)
    potential_poses = xp.concat([list_angles, list_translation], axis=1)
    return potential_poses, precision


def find_angles_grid(
    xp: "array_api_module",
    reconstruction: "Array",
    patches: "Array",
    psf: "Array",
    precision: int = 10,
):
    L = find_L(precision)
    potential_poses, _ = create_poses_grid(
        xp,
        L,
        1,
        symmetry=1,
        dtype=reconstruction.dtype,
        device=get_device(reconstruction),
    )
    potential_poses = xp.broadcast_to(potential_poses, (patches.shape[0],)+potential_poses.shape)
    best_poses, best_errors = convolution_matching_poses(
        reconstruction, patches, psf, potential_poses
    )

    return best_poses, best_errors


def get_refined_values1D_uniform(
    xp: "array_api_module", loc: float, N: int, range: float, **array_kwargs
):
    return xp.linspace(loc - range / 2, loc + range / 2, N, **array_kwargs)


def get_refined_valuesND(
    xp: "array_api_module",
    locs: List[float],
    N: List[int],
    ranges: List[float],
    **kwargs,
):
    n = len(locs)
    assert n == len(N) == len(ranges)
    values_1d = [
        get_refined_values1D_uniform(xp, locs[i], N[i], range=ranges[i], **kwargs)
        for i in range(n)
    ]

    return xp.asarray(list(itertools.product(*values_1d)))


def create_poses_refined(
    poses: "Array",
    ranges: List[float],
    M: List[int],
):
    xp = array_namespace(poses)
    n_poses = 1
    for x in M:
        n_poses *= x
    potential_poses = xp.stack((xp.asarray(poses, copy=True),) * n_poses, axis=1)
    potential_poses[..., 3:] = 0
    for i in range(poses.shape[0]):
        potential_poses[i, :, :3] = get_refined_valuesND(
            xp, poses[i, :3], M, ranges, dtype=poses.dtype, device=get_device(poses)
        )

    return potential_poses


def refine(
    patches: "Array",
    psf: "Array",
    guessed_poses: "Array",
    steps: List[Union[Tuple[int, int], int]],
    ranges: List[float],
    initial_volume: Optional["Array"] = None,
    lambda_: float = 100.0,
    symmetry: int = 1,
    convention: str = "XZX",
    device: Optional["Device"] = None,
    batch_size: Optional[int] = None,
):
    """
    Args:
        patches:
            shape (N, C, D, D, D)
        psf:
            shape (C, d, h, w)
        symmetry: if greater than 1, adds a symmetry constraint.
            In that case, the symmetry axis must be parallel to the X-axis.
            See get_transformation_matrix function docs for details
            about the convention.
    """
    assert len(steps) == len(ranges), "steps and ranges lists should have equal length"
    assert len(steps) > 0, "length of steps and ranges lists should be at least 1"
    assert symmetry >= 1, "symmetry should be an integer greater or equal to 1"
    assert lambda_ > 0, f"lambda should be greater than 1, found {lambda_}"
    refinement_logger.debug("Calling function refine")
    xp = array_namespace(patches, psf, guessed_poses)
    N, C, D, H, W = patches.shape
    assert D == H == W
    host_device = get_device(patches)
    compute_device = device
    array_kwargs = dict(dtype=patches.dtype, device=host_device)
    lambda_ = xp.asarray(lambda_, **array_kwargs)

    if initial_volume is not None:
        initial_volume = interpolate_to_size(
            initial_volume, (D, D, D), multichannel=True
        )
        current_reconstruction = initial_volume
    else:
        guessed_poses_sym = symmetrize_poses(
            guessed_poses, symmetry=symmetry, convention=convention
        )
        guessed_poses_sym = xp.permute_dims(guessed_poses_sym, (1, 0, 2))
        initial_reconstruction = reconstruction_L2(
            patches,
            psf,
            guessed_poses_sym,
            lambda_,
            symmetry=True,
            device=compute_device,
            batch_size=batch_size,
            multichannel=True,
        )
        initial_reconstruction = interpolate_to_size(
            initial_reconstruction, (D, D, D), multichannel=True
        )

        current_reconstruction = initial_reconstruction

    current_poses = guessed_poses

    if refinement_logger.isEnabledFor(logging.DEBUG):
        im = to_numpy(current_reconstruction)
        p = debug.save_image(
            im,
            debug.DEBUG_DIR_REFINEMENT,
            refine,
            "initial-reconstruction",
            multichannel=True,
        )
        refinement_logger.debug("Saving current reconstruction at " + str(p))
        all_recons = [im]
        debug.save_image(to_numpy(psf), debug.DEBUG_DIR_REFINEMENT, refine, "psf", multichannel=True)
    
        for j in range(len(current_poses)):
            refinement_logger.debug(
                f"pose[{j}], found: ["
                + ", ".join(
                    [f"{x:.1f}" for x in to_numpy(current_poses[j]).tolist()]
                )
                + "]",
            )
            patch_j_trans = to_numpy(
                affine_transform(
                    patches[j],
                    xp.astype(
                        get_transform_matrix_from_pose((D, D, D), current_poses[j]),
                        patches.dtype,
                    ),
                    multichannel=True,
                    order=1,
                )
            )
            debug.save_image(
                patch_j_trans,
                debug.DEBUG_DIR_REFINEMENT,
                refine,
                f"step0-patch{j}-pose",
                multichannel=True,
            )
    for i in tqdm(range(len(steps)), desc="refine"):
        refinement_logger.debug(f"STEP {i+1}/{len(steps)}")
        t1 = time.time()
        # Poses estimation
        s = steps[i]
        if ranges[i] == 0 and type(s) is tuple:  # Discretization of the whole sphere
            M_axes, M_rot = s
            potential_poses, (precision_axes, precision_rot) = create_poses_grid(
                xp, M_axes, M_rot, symmetry=symmetry, **array_kwargs
            )
            potential_poses = xp.broadcast_to(potential_poses, (N, M_axes*M_rot, 6))
            refinement_logger.debug(
                "[convolution_matching_poses] Searching the whole grid. "
                f"N_axes={M_axes}, N_rot={M_rot}. "
                f"precision_axes={precision_axes:.2f}°, "
                f"precision_rot={precision_rot:.2f}°"
            )
            
        elif isinstance(s, int):  # Refinement around the current poses
            refinement_logger.debug(
                f"[convolution_matching_poses] Refining the poses. range={ranges[i]}, steps={s}"
            )
            t0 = time.time()
            potential_poses = create_poses_refined(current_poses, [ranges[i]] * 3, [s] * 3)
        else:
            raise ValueError(
                "When range==0, steps should be a tuple. "
                "When range>0, steps should be an int. "
                f"Found range={ranges[i]} and steps={s}"
            )

        t0 = time.time()
        current_poses, _ = convolution_matching_poses(
            current_reconstruction,
            patches,
            psf,
            potential_poses,
            device=compute_device,
            batch_size=batch_size,
        )
        refinement_logger.debug(
            f"[convolution_matching_poses] Done in {time.time()-t0:.3f}s"
        )

        # Reconstruction
        refinement_logger.debug("[reconstruction_L2] Reconstruction")
        t0 = time.time()
        current_poses_sym = symmetrize_poses(
            current_poses, symmetry=symmetry, convention=convention
        )
        current_poses_sym = xp.permute_dims(current_poses_sym, (1, 0, 2))
        current_reconstruction = reconstruction_L2(
            patches,
            psf,
            current_poses_sym,
            lambda_,
            symmetry=True,
            device=compute_device,
            batch_size=batch_size,
            multichannel=True,
        )
        current_reconstruction = interpolate_to_size(
            current_reconstruction, (D, D, D), multichannel=True
        )
        refinement_logger.debug(f"[reconstruction_L2] Done in {time.time()-t0:.3f}s")

        if refinement_logger.isEnabledFor(
            logging.DEBUG
        ):  # .cpu() causes host-device sync
            for j in range(len(current_poses)):
                refinement_logger.debug(
                    f"pose[{j}], found: ["
                    + ", ".join(
                        [f"{x:.1f}" for x in to_numpy(current_poses[j]).tolist()]
                    )
                    + "]",
                )
                patch_j_trans = to_numpy(
                    affine_transform(
                        patches[j],
                        xp.astype(
                            get_transform_matrix_from_pose((D, D, D), current_poses[j]),
                            patches.dtype,
                        ),
                        multichannel=True,
                        order=1,
                    )
                )
                debug.save_image(
                    patch_j_trans,
                    debug.DEBUG_DIR_REFINEMENT,
                    refine,
                    f"step{i+1}-patch{j}-pose",
                    multichannel=True,
                )

            im = to_numpy(current_reconstruction)
            p = debug.save_image(
                im, debug.DEBUG_DIR_REFINEMENT, refine, f"step{i+1}", multichannel=True
            )
            refinement_logger.debug("Saving current reconstruction at " + str(p))
            all_recons.append(im)

        refinement_logger.debug(
            f"STEP {i+1}/{len(steps)} done in {time.time()-t1:.3f}s"
        )

    if refinement_logger.isEnabledFor(logging.DEBUG):
        p = debug.save_image(
            np.stack(all_recons, axis=0),
            debug.DEBUG_DIR_REFINEMENT,
            refine,
            "all-steps",
            sequence=True,
            multichannel=True,
        )
        refinement_logger.debug("Saving all reconstructions at " + str(p))

    return current_reconstruction, current_poses
