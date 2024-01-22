"""Some functions from this file were translated from code written by Denis Fortun"""

import logging
import math
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

import spfluo.utils.debug as debug
from spfluo.utils._torch_functions.volume import fftn, pad_to_size
from spfluo.utils.memory import split_batch_func
from spfluo.utils.transform import get_transform_matrix, symmetrize_poses
from spfluo.utils.volume import (
    affine_transform,
    discretize_sphere_uniformly,
    interpolate_to_size,
    phase_cross_correlation,
)

refinement_logger = logging.getLogger("spfluo.refinement")


def affine_transform_wrapper(volumes, poses, inverse=False):
    H = get_transform_matrix(
        volumes.shape[1:], poses[:, :3], poses[:, 3:], convention="XZX", degrees=True
    ).type(volumes.dtype)
    if not inverse:  # scipy's affine_transform do inverse transform by default
        torch.linalg.inv(H, out=H)
    return affine_transform(volumes, H, order=1, prefilter=False, batch=True)


def reconstruction_L2(
    volumes: torch.Tensor,
    psf: torch.Tensor,
    poses: torch.Tensor,
    lambda_: torch.Tensor,
    batch: bool = False,
    symmetry: bool = False,
) -> torch.Tensor:
    """Reconstruct a particule from volumes and their poses.
    M reconstructions can be done at once.

    Args:
        volumes (torch.Tensor): stack of N 3D images of shape (N, D, D, D)
        psf (torch.Tensor) : 3D image of shape (d, h, w)
        poses (torch.Tensor):
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
        lambda_ (torch.Tensor): regularization parameters of shape ((M),)
        batch (bool): do M reconstructions at once
            poses must be of shape (M, (k), N, 6), the k dim is optional
        symmetry (bool): use a k-degree symmetry
            poses must be of shape ((M), k, N, 6), the M dim is optional

    Returns:
        recon (torch.Tensor):
            reconstruction(s) of shape ((M), D, D, D) or ((M), D+1, D+1, D+1)
    """
    refinement_logger.info("Calling function reconstruction_L2")
    device = volumes.device
    N, D, H, W = volumes.size(-4), volumes.size(-3), volumes.size(-2), volumes.size(-1)
    assert D == H == W
    d, h, w = psf.size(-3), psf.size(-2), psf.size(-1)
    refinement_logger.info(
        "Arguments:" f" {N} volumes of size {D}x{D}x{D}" f" PSF of size {d}x{h}x{w}"
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
        poses = poses.unsqueeze(0)
        lambda_ = lambda_.view(1)

    if symmetry:
        assert poses.ndim == 4
        k = poses.shape[1]
        refinement_logger.info(f"Symmetry enabled, of degree k={k}")
    else:
        k = 1
        poses = poses.unsqueeze(1)  # shape (1, 1, N, 6)

    new_poses = torch.clone(poses)
    if D % 2 == 0:
        # pad by 1 pixel on the right
        volumes = pad_to_size(volumes, np.asarray((N, D + 1, D + 1, D + 1)))
        assert volumes.shape == (N, D + 1, D + 1, D + 1)
        D = D + 1
        new_poses[..., 3:] -= 0.5
        refinement_logger.info(f"Reshaped volumes to odd size {D}x{D}x{D}")
        resize = True
    else:
        resize = False

    psf = interpolate_to_size(psf, (D, D, D))

    num = torch.zeros((M, D, D, D), dtype=torch.complex64, device=device)
    den = torch.zeros_like(num)

    dxyz = torch.zeros((3, 2, 2, 2), device=volumes.device)
    dxyz[0, 0, 0, 0] = 1
    dxyz[0, 1, 0, 0] = -1
    dxyz[1, 0, 0, 0] = 1
    dxyz[1, 0, 1, 0] = -1
    dxyz[2, 0, 0, 0] = 1
    dxyz[2, 0, 0, 1] = -1

    dxyz_padded = pad_to_size(dxyz, (3, D, D, D))
    DtD = (torch.fft.fftn(dxyz_padded, dim=(1, 2, 3)).abs() ** 2).sum(dim=0)
    den += lambda_[:, None, None, None] * DtD
    del DtD

    poses_psf = torch.zeros_like(new_poses)
    poses_psf[..., :3] = new_poses[..., :3]

    for (start1, end1), (start2, end2) in split_batch_func(
        "reconstruction_L2", volumes, psf, new_poses, lambda_
    ):
        size_batch = (end1 - start1) * (end2 - start2)
        y = volumes.unsqueeze(1).repeat(size_batch, 1, 1, 1, 1)
        y = affine_transform_wrapper(
            y.view(size_batch * N, D, D, D),
            new_poses[start1:end1, start2:end2, :, :].reshape(size_batch * N, 6),
            inverse=True,
        ).view(end1 - start1, end2 - start2, N, D, D, D)
        y = y.type(torch.complex64)

        H_ = psf.unsqueeze(0).repeat(N * size_batch, 1, 1, 1)
        H_ = affine_transform_wrapper(
            H_,
            poses_psf[start1:end1, start2:end2, :, :].reshape(size_batch * N, 6),
            inverse=True,
        ).view(end1 - start1, end2 - start2, N, D, D, D)
        H_ = H_.type(torch.complex64)

        fftn(H_, dim=(-3, -2, -1), out=H_)

        # Compute numerator
        fftn(torch.fft.fftshift(y, dim=(-3, -2, -1)), dim=(-3, -2, -1), out=y)
        y = H_.conj() * y
        num[start1:end1] += torch.sum(y, dim=(-5, -4))  # reduce symmetry and N dims

        # Compute denominator
        torch.abs(torch.mul(H_.conj(), H_, out=H_), out=H_)
        den[start1:end1] += torch.sum(H_, dim=(-5, -4))
        del H_

        torch.cuda.empty_cache()

    torch.fft.ifftn(num.div_(den), dim=(-3, -2, -1), out=num)
    recon = num.real
    del num
    recon = torch.clamp(recon, min=0)

    if resize:
        recon = interpolate_to_size(recon, (D - 1, D - 1, D - 1), batch=True)

    if not batch:
        recon = recon[0]

    if refinement_logger.isEnabledFor(logging.DEBUG):
        p = debug.save_image(
            recon.cpu().numpy(),
            debug.DEBUG_DIR_REFINEMENT,
            reconstruction_L2,
            "reconstruction",
            sequence=batch,
        )
        refinement_logger.debug("Saving reconstruction(s) at " + str(p))

    return recon


def convolution_matching_poses_grid(
    reference: torch.Tensor,
    volumes: torch.Tensor,
    psf: torch.Tensor,
    poses_grid: torch.Tensor,
) -> Tuple[torch.Tensor]:
    """Find the best pose from a list of poses for each volume
    Params:
        reference (torch.Tensor) : reference 3D image of shape (D, H, W)
        volumes (torch.Tensor) : volumes to match of shape (N, D, H, W)
        psf (torch.Tensor): 3D PSF of shape (d, h, w)
        poses_grid (torch.Tensor): poses to test of shape (M, 6)
    Returns:
        best_poses (torch.Tensor): best poses for each volume of shape (N, 6)
        best_errors (torch.Tensor): dftRegistration error associated to each pose (N,)
    """
    # Shapes
    M, d = poses_grid.shape
    N, D, H, W = volumes.shape

    # PSF
    h = torch.fft.fftn(torch.fft.fftshift(interpolate_to_size(psf, (D, H, W))))

    shifts = torch.empty((N, M, 3))
    errors = torch.empty((N, M))
    for (start1, end1), (start2, end2) in split_batch_func(
        "convolution_matching_poses_grid", reference, volumes, psf, poses_grid
    ):
        potential_poses_minibatch = poses_grid[start2:end2]

        # Volumes to frequency space
        volumes_freq = torch.fft.fftn(volumes[start1:end1], dim=(1, 2, 3))

        # Rotate the reference
        reference_minibatch = reference.repeat(end2 - start2, 1, 1, 1)
        reference_minibatch = affine_transform_wrapper(  # TODO: inefficient
            reference_minibatch, potential_poses_minibatch  # should leverage pytorch
        )  # multichannel grid_sample
        reference_minibatch = h * torch.fft.fftn(reference_minibatch, dim=(1, 2, 3))

        # Registration
        sh, err, _ = phase_cross_correlation(
            reference_minibatch[None],
            volumes_freq[:, None],
            nb_spatial_dims=3,
            normalization=None,
            upsample_factor=10,
            space="fourier",
        )
        sh = torch.stack(list(sh), dim=-1)

        errors[start1:end1, start2:end2] = err
        shifts[start1:end1, start2:end2] = sh

        del volumes_freq, reference_minibatch, err, sh
        torch.cuda.empty_cache()

    best_errors, best_indices = torch.min(errors, dim=1)
    best_poses = poses_grid[best_indices]
    best_poses[:, 3:] = -shifts[np.arange(N), best_indices]

    return best_poses, best_errors


def convolution_matching_poses_refined(
    reference: torch.Tensor,
    volumes: torch.Tensor,
    psf: torch.Tensor,
    potential_poses: torch.Tensor,
) -> Tuple[torch.Tensor]:
    """Find the best pose from a list of poses for each volume.
    There can be a different list of pose for each volume.
    Params:
        reference (torch.Tensor) : reference 3D image of shape (D, H, W)
        volumes (torch.Tensor) : volumes to match of shape (N, D, H, W)
        psf (torch.Tensor): 3D PSF of shape (d, h, w)
        potential_poses (torch.Tensor): poses to test of shape (N, M, 6)
    Returns:
        best_poses (torch.Tensor): best poses for each volume of shape (N, 6)
        best_errors (torch.Tensor): dftRegistration error associated to each pose (N,)
    """
    # Shapes
    N1, M, d = potential_poses.shape
    N, D, H, W = volumes.shape
    assert N == N1

    # PSF
    h = torch.fft.fftn(torch.fft.fftshift(interpolate_to_size(psf, (D, H, W))))

    shifts = torch.empty((N, M, 3))
    errors = torch.empty((N, M))
    for (start1, end1), (start2, end2) in split_batch_func(
        "convolution_matching_poses_refined", reference, volumes, psf, potential_poses
    ):
        minibatch_size = (end1 - start1) * (end2 - start2)
        potential_poses_minibatch = potential_poses[
            start1:end1, start2:end2
        ].contiguous()

        # Volumes to Fourier space
        volumes_freq = torch.fft.fftn(volumes[start1:end1], dim=(1, 2, 3))

        # Rotate the reference
        reference_minibatch = reference.repeat(minibatch_size, 1, 1, 1)
        reference_minibatch = affine_transform_wrapper(
            reference_minibatch,
            potential_poses_minibatch.view(minibatch_size, d),
        ).view(end1 - start1, end2 - start2, D, H, W)
        reference_minibatch = h * torch.fft.fftn(reference_minibatch, dim=(2, 3, 4))

        # Registration
        sh, err, _ = phase_cross_correlation(
            reference_minibatch,
            volumes_freq[:, None],
            nb_spatial_dims=3,
            normalization=None,
            upsample_factor=10,
            space="fourier",
        )
        sh = torch.stack(list(sh), dim=-1)

        errors[start1:end1, start2:end2] = err
        shifts[start1:end1, start2:end2] = sh

        del volumes_freq, reference_minibatch, err, sh
        torch.cuda.empty_cache()

    errors, best_indices = torch.min(errors, dim=1)
    best_poses = potential_poses[np.arange(N), best_indices]
    best_poses[:, 3:] = -shifts[np.arange(N), best_indices]
    return best_poses, errors


def find_L(precision):
    return math.ceil(((360 / precision) ** 2) / torch.pi)


def create_poses_grid(M_axes, M_rot, symmetry=1, **tensor_kwargs):
    (theta, phi, psi), precision = discretize_sphere_uniformly(
        torch, M_axes, M_rot, product=True, symmetry=symmetry, **tensor_kwargs
    )
    list_angles = torch.stack([theta, phi, psi], dim=-1)
    M = list_angles.shape[0]
    list_translation = torch.zeros((M, 3), **tensor_kwargs)
    potential_poses = torch.cat([list_angles, list_translation], dim=1)
    return potential_poses, precision


def find_angles_grid(reconstruction, patches, psf, precision=10):
    L = find_L(precision)
    potential_poses, _ = create_poses_grid(
        L, 1, symmetry=1, dtype=reconstruction.dtype, device=reconstruction.device
    )
    best_poses, best_errors = convolution_matching_poses_grid(
        reconstruction, patches, psf, potential_poses
    )

    return best_poses, best_errors


def get_refined_values1D_uniform(loc: float, N: int, range: float, **tensor_kwargs):
    return torch.linspace(loc - range / 2, loc + range / 2, N, **tensor_kwargs)


def get_refined_values1D_gaussian(
    loc, N, sigma=10, range=0.8, device=None, dtype=torch.float32
):
    d = torch.distributions.normal.Normal(loc, sigma)
    step = range / N
    lowest = 0.5 - torch.floor(N / 2) * step
    highest = 0.5 + torch.ceil(N / 2) * step
    return d.icdf(torch.arange(lowest, highest, step, device=device, dtype=dtype))


def get_refined_valuesND(
    locs: List[float],
    N: List[int],
    ranges: List[float],
    method: str = "uniform",
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    n = len(locs)
    assert n == len(N) == len(ranges)
    if method == "uniform":
        values_1d = [
            get_refined_values1D_uniform(locs[i], N[i], range=ranges[i], **kwargs)
            for i in range(n)
        ]
    elif method == "gaussian":
        values_1d = [
            get_refined_values1D_gaussian(
                locs[i], N[i], sigma=sigmas[i], range=ranges[i], **kwargs
            )
            for i in range(n)
        ]

    return torch.cartesian_prod(*values_1d)


def create_poses_refined(
    poses: torch.Tensor, ranges: List[float], M: List[int], **kwargs
):
    potential_poses = torch.clone(poses).unsqueeze(1).repeat(1, np.prod(M), 1)
    for i in range(poses.size(0)):
        potential_poses[i, :, :3] = get_refined_valuesND(
            poses[i, :3], M, ranges, **kwargs
        )

    return potential_poses


def refine_poses(
    reconstruction: torch.Tensor,
    patches: torch.Tensor,
    psf: torch.Tensor,
    guessed_poses: torch.Tensor,
    range: float,
    steps: int,
) -> Tuple[torch.Tensor]:
    device = reconstruction.device
    dtype = reconstruction.dtype
    potential_poses = create_poses_refined(
        guessed_poses, [range] * 3, [steps] * 3, dtype=dtype, device=device
    )

    best_poses, best_errors = convolution_matching_poses_refined(
        reconstruction, patches, psf, potential_poses
    )

    return best_poses, best_errors


def refine(
    patches: torch.Tensor,
    psf: torch.Tensor,
    guessed_poses: torch.Tensor,
    steps: List[Union[Tuple[int, int], int]],
    ranges: List[float],
    initial_volume: Optional[torch.Tensor] = None,
    lambda_: float = 100.0,
    symmetry: int = 1,
    convention: str = "XZX",
):
    """
    Args:
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
    tensor_kwargs = dict(dtype=patches.dtype, device=patches.device)
    lambda_ = torch.tensor(lambda_, **tensor_kwargs)

    if initial_volume is not None:
        initial_volume = interpolate_to_size(initial_volume, patches[0].shape)
        current_reconstruction = initial_volume
    else:
        guessed_poses_sym = symmetrize_poses(
            guessed_poses, symmetry=symmetry, convention=convention
        )
        guessed_poses_sym = torch.permute(guessed_poses_sym, (1, 0, 2)).contiguous()
        initial_reconstruction = reconstruction_L2(
            patches, psf, guessed_poses_sym, lambda_, symmetry=True
        )
        initial_reconstruction = interpolate_to_size(
            initial_reconstruction, patches[0].shape
        )

        current_reconstruction = initial_reconstruction

    if refinement_logger.isEnabledFor(logging.DEBUG):
        im = current_reconstruction.cpu().numpy()
        p = debug.save_image(
            im, debug.DEBUG_DIR_REFINEMENT, refine, "initial-reconstruction"
        )
        refinement_logger.debug("Saving current reconstruction at " + str(p))
        all_recons = [im]

    current_poses = guessed_poses
    for i in tqdm(range(len(steps)), desc="refine"):
        refinement_logger.debug(f"STEP {i+1}/{len(steps)}")
        t1 = time.time()
        # Poses estimation
        s = steps[i]
        if ranges[i] == 0 and type(s) is tuple:  # Discretization of the whole sphere
            M_axes, M_rot = s
            potential_poses, (precision_axes, precision_rot) = create_poses_grid(
                M_axes, M_rot, symmetry=symmetry, **tensor_kwargs
            )
            refinement_logger.debug(
                "[convolution_matching_poses_grid] Searching the whole grid. "
                f"N_axes={M_axes}, N_rot={M_rot}. "
                f"precision_axes={precision_axes:.2f}°, "
                f"precision_rot={precision_rot:.2f}°"
            )
            t0 = time.time()
            current_poses, _ = convolution_matching_poses_grid(
                current_reconstruction, patches, psf, potential_poses
            )
            refinement_logger.debug(
                f"[convolution_matching_poses_grid] Done in {time.time()-t0:.3f}s"
            )
        elif isinstance(s, int):  # Refinement around the current poses
            refinement_logger.debug(
                f"[refine_poses] Refining the poses. range={ranges[i]}, steps={s}"
            )
            t0 = time.time()
            current_poses, _ = refine_poses(
                current_reconstruction, patches, psf, current_poses, ranges[i], s
            )
            refinement_logger.debug(f"[refine_poses] Done in {time.time()-t0:.3f}s")
        else:
            raise ValueError(
                "When range==0, steps should be a tuple. "
                "When range>0, steps should be an int. "
                f"Found range={ranges[i]} and steps={s}"
            )

        # Reconstruction
        refinement_logger.debug("[reconstruction_L2] Reconstruction")
        t0 = time.time()
        current_poses_sym = symmetrize_poses(
            current_poses, symmetry=symmetry, convention=convention
        )
        current_poses_sym = torch.permute(current_poses_sym, (1, 0, 2)).contiguous()
        current_reconstruction = reconstruction_L2(
            patches, psf, current_poses_sym, lambda_, symmetry=True
        )
        current_reconstruction = interpolate_to_size(
            current_reconstruction, patches[0].shape
        )
        refinement_logger.debug(f"[reconstruction_L2] Done in {time.time()-t0:.3f}s")

        if refinement_logger.isEnabledFor(
            logging.DEBUG
        ):  # .cpu() causes host-device sync
            for j in range(len(current_poses)):
                refinement_logger.debug(
                    f"pose[{j}], found: ["
                    + ", ".join([f"{x:.1f}" for x in current_poses[j].cpu().tolist()])
                    + "]",
                )
            im = current_reconstruction.cpu().numpy()
            p = debug.save_image(im, debug.DEBUG_DIR_REFINEMENT, refine, f"step{i+1}")
            refinement_logger.debug("Saving current reconstruction at " + str(p))
            all_recons.append(im)

        refinement_logger.debug(
            f"STEP {i+1}/{len(steps)} done in {time.time()-t1:.3f}s"
        )

        torch.cuda.empty_cache()

    if refinement_logger.isEnabledFor(logging.DEBUG):
        p = debug.save_image(
            np.stack(all_recons, axis=0),
            debug.DEBUG_DIR_REFINEMENT,
            refine,
            "all-steps",
            sequence=True,
        )
        refinement_logger.debug("Saving all reconstructions at " + str(p))

    return current_reconstruction, current_poses


def first_reconstruction(patches, views, poses, psf, step=10):
    errors = []
    recons = []
    lambda_ = 5e-2
    poses_known = torch.zeros_like(poses)
    poses_known[views == 0, 1:3] = 0
    poses_known[views == 1, 1] = 90
    deltas = torch.arange(0, 360, step, dtype=patches.dtype)
    for delta in deltas:
        poses_known[views == 1, 2] = poses[views == 1, 2] + delta

        # reconstruction L2
        mask_top_side = torch.logical_or(
            torch.as_tensor(views == 0), torch.as_tensor(views == 1)
        )
        recon_noised = reconstruction_L2(
            patches[mask_top_side], psf, poses_known[mask_top_side], lambda_
        )
        recons.append(recon_noised)

        # compute error
        N = patches[mask_top_side].shape[0]
        recon_noised_transformed = affine_transform_wrapper(
            recon_noised[None].repeat(N, 1, 1, 1), poses_known[mask_top_side]
        )
        error = (
            ((recon_noised_transformed - patches[mask_top_side]) ** 2)
            .view(N, -1)
            .sum(dim=1)
            ** 0.5
        ).sum() / N
        errors.append(error)

    recons = torch.stack(recons)
    errors = torch.stack(errors)
    i = errors.argmin()

    return deltas[i], recons[i], errors[i]
