"""Some functions from this file were translated from code written by Denis Fortun"""

import logging
import math
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from spfluo.utils import (
    affine_transform,
    discretize_sphere_uniformly,
    fftn,
    phase_cross_correlation,
)
from spfluo.utils.memory import split_batch_func
from spfluo.utils.transform import get_transform_matrix
from spfluo.utils.volume import interpolate_to_size

refinement_logger = logging.getLogger("spfluo.refinement")
if refinement_logger.isEnabledFor(logging.DEBUG):
    from spfluo.utils.debug import DEBUG_DIR, save_image

    DEBUG_DIR_REFINEMENT = DEBUG_DIR / __name__
    DEBUG_DIR_REFINEMENT.mkdir(parents=True, exist_ok=False)


def affine_transform_wrapper(volumes, poses, inverse=False):
    H = get_transform_matrix(
        volumes.shape[1:], poses[:, :3], poses[:, 3:], convention="XZX", degrees=True
    ).type(volumes.dtype)
    if not inverse:  # scipy's affine_transform do inverse transform by default
        torch.linalg.inv(H, out=H)
    return affine_transform(volumes, H, order=1, prefilter=False, batch=True)


def reconstruction_L2(
    volumes: torch.Tensor, psf: torch.Tensor, poses: torch.Tensor, lambda_: torch.Tensor
) -> torch.Tensor:
    """Reconstruct a particule from volumes and their poses.
    M reconstructions can be done at once.

    Args:
        volumes (torch.Tensor): stack of N 3D images of shape (N, D, H, W)
        psf (torch.Tensor) : 3D image of shape (d, h, w)
        poses (torch.Tensor):
            stack(s) of N poses of shape (N, 6) or (M, N, 6).
            Euler angles in the 'zxz' convention in degrees
            Translation vector tz, ty, tx
        lambda_ (torch.Tensor): regularization parameters of shape () or (M,)

    Returns:
        recon (torch.Tensor): reconstruction(s) of shape (D, H, W) or (M, D, H, W)
        den (torch.Tensor): something of shape (D, H, W) or (M, D, H, W)
    """
    dtype, device = volumes.dtype, volumes.device
    N, D, H, W = volumes.size(-4), volumes.size(-3), volumes.size(-2), volumes.size(-1)
    d, h, w = psf.size(-3), psf.size(-2), psf.size(-1)
    batch_dims = torch.as_tensor(poses.shape[:-2])
    batched = True
    if len(batch_dims) == 0:
        poses = poses.unsqueeze(0)
        lambda_ = lambda_.view(1)
        batched = False
        batch_dims = (1,)

    recon = torch.empty(tuple(batch_dims) + (D, H, W), dtype=dtype, device=device)
    den = torch.empty_like(recon)

    dxyz = torch.zeros((3, 2, 2, 2), device=volumes.device)
    dxyz[0, 0, 0, 0] = 1
    dxyz[0, 1, 0, 0] = -1
    dxyz[1, 0, 0, 0] = 1
    dxyz[1, 0, 1, 0] = -1
    dxyz[2, 0, 0, 0] = 1
    dxyz[2, 0, 0, 1] = -1

    dxyz_padded = interpolate_to_size(dxyz, (D, H, W), batch=True)
    DtD = (torch.fft.fftn(dxyz_padded, dim=(1, 2, 3)).abs() ** 2).sum(dim=0)

    poses_psf = torch.zeros_like(poses)
    poses_psf[:, :, :3] = poses[:, :, :3]

    for start, end in split_batch_func(
        "reconstruction_L2", volumes, psf, poses, lambda_
    ):
        size_batch = end - start
        y = volumes.unsqueeze(1).repeat(
            size_batch, 1, 1, 1, 1
        )  # shape (size_batch, N, D, H, W)
        y = affine_transform_wrapper(
            y.view(size_batch * N, D, H, W),
            poses[start:end].view(size_batch * N, 6),
            inverse=True,
        ).view(size_batch, N, D, H, W)
        y = y.type(torch.complex64)

        h_ = psf.unsqueeze(0).repeat(N * size_batch, 1, 1, 1)
        h_ = affine_transform_wrapper(
            h_, poses_psf[start:end].view(size_batch * N, 6), inverse=True
        ).view(size_batch, N, d, h, w)
        H_ = interpolate_to_size(h_.view(-1, d, h, w), (D, H, W), batch=True).view(
            size_batch, N, D, H, W
        )
        H_ = H_.type(torch.complex64)
        del h_

        fftn(H_, dim=(-3, -2, -1), out=H_)
        fftn(torch.fft.fftshift(y, dim=(-3, -2, -1)), dim=(-3, -2, -1), out=y)

        torch.mul(H_.conj(), y, out=y)
        torch.abs(torch.mul(H_.conj(), H_, out=H_), out=H_)

        y = torch.mean(y, dim=-4)
        torch.mean(H_, dim=-4, out=den[start:end])
        del H_

        den[start:end] += lambda_[start:end, None, None, None] * DtD
        torch.fft.ifftn(y.div_(den[start:end]), dim=(-3, -2, -1), out=y)
        recon[start:end] = y.real
        torch.clamp(recon[start:end], min=0, out=recon[start:end])

        torch.cuda.empty_cache()

    if not batched:
        recon, den = recon[0], den[0]

    return recon, den


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
        psf (torch.Tensor): 3D PSF of shape (D, H, W)
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
        psf (torch.Tensor): 3D PSF of shape (D, H, W)
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
    lambda_: float = 100.0,
    symmetry: int = 1,
):
    assert len(steps) == len(ranges), "steps and ranges lists should have equal length"
    assert len(steps) > 0, "length of steps and ranges lists should be at least 1"
    assert symmetry >= 1, "symmetry should be an integer greater or equal to 1"
    assert lambda_ > 0, f"lambda should be greater than 1, found {lambda_}"
    refinement_logger.debug("Calling function refine")
    tensor_kwargs = dict(dtype=patches.dtype, device=patches.device)
    lambda_ = torch.tensor(lambda_, **tensor_kwargs)
    initial_reconstruction, _ = reconstruction_L2(patches, psf, guessed_poses, lambda_)

    if refinement_logger.isEnabledFor(logging.DEBUG):
        im = initial_reconstruction.cpu().numpy()
        p = save_image(im, DEBUG_DIR_REFINEMENT, refine, "initial-reconstruction")
        refinement_logger.debug("Saving current reconstruction at " + str(p))
        all_recons = [im]

    current_reconstruction = initial_reconstruction
    current_poses = guessed_poses
    for i in range(len(steps)):
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
        elif type(s) is int:  # Refinement around the current poses
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
        current_reconstruction, _ = reconstruction_L2(
            patches, psf, current_poses, lambda_
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
            p = save_image(im, DEBUG_DIR_REFINEMENT, refine, f"step{i+1}")
            refinement_logger.debug("Saving current reconstruction at " + str(p))
            all_recons.append(im)

        refinement_logger.debug(
            f"STEP {i+1}/{len(steps)} done in {time.time()-t1:.3f}s"
        )

    if refinement_logger.isEnabledFor(logging.DEBUG):
        p = save_image(
            np.stack(all_recons, axis=0),
            DEBUG_DIR_REFINEMENT,
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
        recon_noised, _ = reconstruction_L2(
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
