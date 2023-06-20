"""Some functions from this file were translated from a Matlab project made by Denis Fortun"""

from spfluo.utils import pad_to_size, fftn, dftregistrationND, discretize_sphere_uniformly, affine_transform
from spfluo.utils.memory import split_batch_func

from typing import Tuple
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def affine_transform_wrapper(volumes, poses):
    tensor_kwargs = {"device": volumes.device, "dtype": volumes.dtype}
    matrices = torch.tensor(Rotation.from_euler("ZXZ", poses[:,:3].cpu().numpy(), degrees=True).as_matrix(), **tensor_kwargs)
    return affine_transform(volumes, matrices)

def reconstruction_L2(volumes: torch.Tensor, psf: torch.Tensor, poses: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
    """Reconstruct a particule from volumes and their poses. M reconstructions can be done at once.

    Args:
        volumes (torch.Tensor): stack of N 3D images of shape (N, D, H, W)
        psf (torch.Tensor) : 3D image of shape (d, h, w)
        poses (torch.Tensor): stack(s) of N transform matrices of shape (N, 6) or (M, N, 6).
            Euler angles in the 'zxz' convention in degrees and translation vector tz, ty, tx
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
    
    recon = torch.empty(tuple(batch_dims)+(D,H,W), dtype=dtype, device=device)
    den = torch.empty_like(recon)

    dxyz = torch.zeros((3,2,2,2), device=volumes.device)
    dxyz[0,0,0,0] = 1 
    dxyz[0,1,0,0] = -1
    dxyz[1,0,0,0] = 1 
    dxyz[1,0,1,0] = -1
    dxyz[2,0,0,0] = 1 
    dxyz[2,0,0,1] = -1

    dxyz_padded = pad_to_size(dxyz, (3,) + volumes.shape[-3:])
    DtD = (torch.fft.fftn(dxyz_padded, dim=(1,2,3)).abs()**2).sum(dim=0)

    poses_psf = torch.zeros_like(poses)
    poses_psf[:, :, :3] = poses[:, :, :3]

    for start, end in split_batch_func(
        "reconstruction_L2", volumes, psf, poses, lambda_
    ):
        size_batch = end - start
        y = volumes.unsqueeze(1).repeat(size_batch,1,1,1,1) # shape (size_batch, N, D, H, W)
        y = affine_transform_wrapper(y.view(size_batch*N,D,H,W)[:,None], poses[start:end].view(size_batch*N,6))[:,0].view(size_batch,N,D,H,W)
        y = y.type(torch.complex64)
        
        h_ = psf.unsqueeze(0).repeat(N*size_batch,1,1,1).unsqueeze(1)
        h_ = affine_transform_wrapper(h_, poses_psf[start:end].view(size_batch*N,6))[:,0].view(size_batch,N,d,h,w)
        H_ = pad_to_size(h_, y.size())
        H_ = H_.type(torch.complex64)
        del h_

        fftn(H_, dim=(-3,-2,-1), out=H_)
        fftn(torch.fft.fftshift(y, dim=(-3,-2,-1)), dim=(-3,-2,-1), out=y)
        
        torch.mul(H_.conj(), y, out=y)
        torch.abs(torch.mul(H_.conj(), H_, out=H_), out=H_)
        
        y = torch.mean(y, dim=-4) 
        torch.mean(H_, dim=-4, out=den[start:end])
        del H_
        
        den[start:end] += lambda_[start:end,None,None,None] * DtD
        torch.fft.ifftn(y.div_(den[start:end]), dim=(-3,-2,-1), out=y)
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
    potential_poses: torch.Tensor
) -> Tuple[torch.Tensor]:
    """Find the best pose from a list of poses for each volume
    Params:
        reference (torch.Tensor) : reference 3D image of shape (D, H, W)
        volumes (torch.Tensor) : volumes to match of shape (N, D, H, W)
        psf (torch.Tensor): 3D PSF of shape (D, H, W)
        potential_poses (torch.Tensor): poses to test of shape (M, 6)
    Returns:
        best_poses (torch.Tensor): best poses for each volume of shape (N, 6)
        best_errors (torch.Tensor): dftRegistration error associated to each pose (N,)
    """
    # Shapes
    M, _ = potential_poses.shape
    N, D, H, W = volumes.shape

    # PSF
    h = torch.fft.fftn(torch.fft.fftshift(pad_to_size(psf, (D,H,W))))

    shifts = torch.empty((N,M,3))
    errors = torch.empty((N,M))
    for (start1, end1), (start2, end2) in split_batch_func(
        "convolution_matching_poses_grid", reference, volumes, psf, potential_poses
    ):
        potential_poses_minibatch = potential_poses[start2:end2]

        # Volumes to frequency space
        volumes_freq = torch.fft.fftn(volumes[start1:end1], dim=(1,2,3))

        # Rotate the reference
        reference_minibatch = reference.repeat(end2-start2, 1, 1, 1)
        reference_minibatch = affine_transform_wrapper(reference_minibatch[:,None], potential_poses_minibatch)[:,0]
        reference_minibatch = h * torch.fft.fftn(reference_minibatch, dim=(1,2,3))

        # Registration
        err, sh = dftregistrationND(reference_minibatch[None], volumes_freq[:,None], nb_spatial_dims=3)
        sh = torch.stack(list(sh), dim=-1)
        
        errors[start1:end1, start2:end2] = err
        shifts[start1:end1, start2:end2] = sh
        
        del volumes_freq, reference_minibatch, err, sh
        torch.cuda.empty_cache()
    
    best_errors, best_indices = torch.min(errors, dim=1)
    best_poses = potential_poses[best_indices]
    best_poses[:, 3:] = shifts[np.arange(N), best_indices]
    
    return best_poses, best_errors


def convolution_matching_poses_refined(
    reference: torch.Tensor,
    volumes: torch.Tensor,
    psf: torch.Tensor,
    potential_poses: torch.Tensor
) -> Tuple[torch.Tensor]:
    """Find the best pose from a list of poses for each volume. There can be a different list of pose
    for each volume.
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
    assert N==N1

    # PSF
    h = torch.fft.fftn(torch.fft.fftshift(pad_to_size(psf, (D,H,W))))

    shifts = torch.empty((N,M,3))
    errors = torch.empty((N,M))
    for (start1, end1), (start2, end2) in split_batch_func(
        "convolution_matching_poses_refined", reference, volumes, psf, potential_poses
    ):
        minibatch_size = (end1-start1)*(end2-start2)
        potential_poses_minibatch = potential_poses[start1:end1, start2:end2].contiguous()

        # Volumes to Fourier space
        volumes_freq = torch.fft.fftn(volumes[start1:end1], dim=(1,2,3))

        # Rotate the reference
        reference_minibatch = reference.repeat(minibatch_size, 1, 1, 1)
        reference_minibatch = affine_transform_wrapper(reference_minibatch[:,None], potential_poses_minibatch.view(minibatch_size,d))[:,0].view(end1-start1,end2-start2,D,H,W)
        reference_minibatch = h * torch.fft.fftn(reference_minibatch, dim=(2,3,4))

        # Registration
        err, sh = dftregistrationND(reference_minibatch, volumes_freq[:,None], nb_spatial_dims=3)
        sh = torch.stack(list(sh), dim=-1)

        errors[start1:end1, start2:end2] = err
        shifts[start1:end1, start2:end2] = sh

        del volumes_freq, reference_minibatch, err, sh
        torch.cuda.empty_cache()
    
    errors, best_indices = torch.min(errors, dim=1)
    best_poses = potential_poses[np.arange(N), best_indices]
    best_poses[:, 3:] = shifts[np.arange(N), best_indices]
    return best_poses, errors

def find_L(precision):
    return math.ceil(((360 / precision) ** 2) / torch.pi)


def create_poses_grid(L, dtype=torch.float64, device=None):
    theta, phi, _, precision = discretize_sphere_uniformly(L, 0, dtype, device)
    list_angles = torch.stack([torch.zeros_like(theta), phi, theta], dim=-1)
    M = list_angles.shape[0]
    list_translation = torch.zeros((M,3), dtype=dtype, device=device)
    potential_poses = torch.cat([list_angles,list_translation], dim=1)
    return potential_poses, precision


def find_angles_grid(reconstruction, patches, psf, precision=10):
    L = find_L(precision)
    potential_poses, _ = create_poses_grid(L, reconstruction.dtype, reconstruction.device)

    best_poses, best_errors = convolution_matching_poses_grid(reconstruction, patches, psf, potential_poses)

    return best_poses, best_errors


def get_refined_values1D(loc, N, sigma=10, range=0.8, device=None, dtype=torch.float32):
    d = torch.distributions.normal.Normal(loc, sigma)
    step = range / N
    lowest = 0.5 - torch.floor(N/2) * step
    highest = 0.5 + torch.ceil(N/2) * step
    return d.icdf(torch.arange(lowest, highest, step, device=device, dtype=dtype))


def get_refined_valuesND(locs, N, sigmas, ranges, **kwargs):
    n = len(locs)
    values_1d = [get_refined_values1D(locs[i], int(N**(1/n)), sigma=sigmas[i], range=ranges[i], **kwargs) for i in range(n)]

    return torch.cartesian_prod(*values_1d)


def create_refined_poses(poses, M, **kwargs):
    potential_poses = torch.clone(poses).unsqueeze(1).repeat(1,M,1)
    for i in range(poses.size(0)):
        pose = poses[i]
        refined_angles = get_refined_valuesND(poses[[1,2]], M, [5,5], [0.9,0.9], **kwargs)
        potential_poses[i, :, [1,2]] = refined_angles
    
    return potential_poses


def refine_poses(reconstruction, patches, psf, guessed_poses, M=20):
    device = reconstruction.device
    dtype = reconstruction.dtype
    potential_poses = create_refined_poses(guessed_poses, M, dtype=dtype, device=device)

    best_poses, best_errors = convolution_matching_poses_refined(reconstruction, patches, psf, potential_poses, max_batch=8)

    return best_poses, best_errors


def distance_poses(p1, p2):
    """ Compute the rotation distance and the euclidean distance between p1 and p2.
    Parameters:
        p1, p2 : torch.Tensor of shape (..., 6). Must be broadcastable. Represents poses (theta,psi,gamma,tz,ty,tx).
    Returns:
        distances : Tuple[torch.Tensor] of shape broadcasted dims.
    """
    # Rotation distance
    rot1, rot2 = p1[...,:3], p2[...,:3]
    rot1 = Rotation.from_euler("ZXZ", rot1.cpu().numpy().reshape(-1,3), degrees=True)
    rot2 = Rotation.from_euler("ZXZ", rot2.cpu().numpy().reshape(-1,3), degrees=True)
    points1 = rot1.apply(np.array([0,0,1]), inverse=False).reshape(p1.shape[:-1]+(3,))
    points2 = rot2.apply(np.array([0,0,1]), inverse=False).reshape(p2.shape[:-1]+(3,))
    angle = np.arccos((points1 * points2).sum(axis=-1)) * 180 / np.pi
    angle = torch.as_tensor(angle, dtype=p1.dtype, device=p1.device)

    # Euclidian distance
    t1, t2 = p1[...,3:], p2[...,3:]
    d = ((t1 - t2)**2).sum(dim=-1) ** 0.5

    return angle, d


def first_reconstruction(patches, views, poses, psf, step=10):
    errors = []
    recons= []
    lambda_ = 5e-2
    poses_known = torch.zeros_like(poses)
    poses_known[views==0, 1:3] = 0
    poses_known[views==1, 1] = 90
    deltas = torch.arange(0, 360, step, dtype=patches.dtype)
    for delta in deltas:
        poses_known[views==1, 2] = poses[views==1, 2] + delta

        # reconstruction L2
        mask_top_side = torch.logical_or(torch.as_tensor(views==0), torch.as_tensor(views==1))
        recon_noised, _ = reconstruction_L2(patches[mask_top_side], psf, poses_known[mask_top_side], lambda_)
        recons.append(recon_noised)
        
        # compute error
        N = patches[mask_top_side].shape[0]
        recon_noised_transformed = affine_transform_wrapper(recon_noised[None,None].repeat(N,1,1,1,1), poses_known[mask_top_side])[:,0]
        error = (((recon_noised_transformed - patches[mask_top_side])**2).view(N,-1).sum(dim=1)**0.5).sum() / N
        errors.append(error)

    recons = torch.stack(recons)
    errors = torch.stack(errors)
    i = errors.argmin()

    return deltas[i], recons[i], errors[i]