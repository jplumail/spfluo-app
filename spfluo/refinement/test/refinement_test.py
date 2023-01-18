from spfluo.utils.loading import loadmat
from spfluo.refinement import convolution_matching_poses_refined, convolution_matching_poses_grid, reconstruction_L2

import os
import torch
import numpy as np


##########################
# Test reconstruction_L2 #
##########################

def test_shapes_reconstruction_L2():
    N, D, H, W = 100, 32, 32, 32
    volumes = torch.randn((N, D, H, W))
    psf = torch.randn((D, H, W))
    poses = torch.randn((N, 6))
    lambda_ = torch.tensor(1.)
    recon, den = reconstruction_L2(volumes, psf, poses, lambda_)
    
    assert recon.shape == (D, H, W)
    assert den.shape == (D, H, W)


def test_parallel_reconstruction_L2():
    batch_dims = (5,5,)
    N, D, H, W = 100, 32, 32, 32
    volumes = torch.randn(batch_dims+(N, D, H, W))
    psf = torch.randn(batch_dims+(D, H, W))
    poses = torch.randn(batch_dims+(N, 6))
    lambda_ = torch.randn(batch_dims)
    recon, _ = reconstruction_L2(volumes, psf, poses, lambda_)
    recon2 = torch.stack([torch.stack([reconstruction_L2(vv, pp, ppoo, ll)[0] for vv, pp, ppoo, ll in zip(v,p,po,l)]) for v, p, po, l in zip(volumes, psf, poses, lambda_)])

    assert recon.shape == batch_dims+(D, H, W)
    assert torch.isclose(recon, recon2).all()


#############################
# Test convolution_matching #
#############################

def test_memory_convolution_matching_poses_grid():
    device = 'cuda'
    D = 32
    for N in [1, 10, 1000, 1500, 5000]:
        N = int(N)
        M = int(10000 / N**0.5)
        reference = torch.randn((D, D, D), device=device)
        volumes = torch.randn((N, D, D, D), device=device)
        psf = torch.randn((D, D, D), device=device)
        potential_poses = torch.randn((M, 6), device=device)

        best_poses, errors = convolution_matching_poses_grid(reference, volumes, psf, potential_poses)

        assert best_poses.shape == (N, 6)
        assert errors.shape == (N,)


def test_shapes_convolution_matching_poses_grid():
    M, d = 5, 6
    N, D, H, W = 100, 32, 32, 32
    reference = torch.randn((D, H, W))
    volumes = torch.randn((N, D, H, W))
    psf = torch.randn((D, H, W))
    potential_poses = torch.randn((M, d))

    best_poses, errors = convolution_matching_poses_grid(reference, volumes, psf, potential_poses)

    assert best_poses.shape == (N, d)
    assert errors.shape == (N,)


def test_matlab_convolution_matching_poses_refined():
    as_tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device='cuda')

    # Load Matlab data
    data_path = os.path.join(os.path.dirname(__file__), "data", "convolution_matching")
    potential_poses_ = loadmat(os.path.join(data_path,"bigListPoses.mat"))["bigListPoses"]
    volumes = np.stack(loadmat(os.path.join(data_path,"inVols.mat"))["inVols"][:,0]).transpose(0,3,2,1)
    best_poses_matlab = loadmat(os.path.join(data_path,"posesNew.mat"))["posesNew"][:,[0,1,2,5,3,4]]
    best_poses_matlab[:, 3:] *= -1
    psf = loadmat(os.path.join(data_path,"psf.mat"))["psf"].transpose(2,1,0)
    reference = loadmat(os.path.join(data_path,"recon.mat"))["recon1"].transpose(2,1,0)

    potential_poses_, volumes, best_poses_matlab, psf, reference = map(
        as_tensor, [potential_poses_, volumes, best_poses_matlab, psf, reference]
    )

    N, M, _ = potential_poses_.shape
    potential_poses = as_tensor(torch.zeros((N, M, 6)))
    potential_poses[:, :, :3] = potential_poses_

    best_poses, _ = convolution_matching_poses_refined(reference, volumes, psf, potential_poses)

    eps = 1e-2
    assert ((best_poses - best_poses_matlab) < eps).all()


def test_shapes_convolution_matching_poses_refined():
    M, d = 5, 6
    N, D, H, W = 100, 32, 32, 32
    reference = torch.randn((D, H, W))
    volumes = torch.randn((N, D, H, W))
    psf = torch.randn((D, H, W))
    potential_poses = torch.randn((N, M, d))

    best_poses, errors = convolution_matching_poses_refined(reference, volumes, psf, potential_poses)

    assert best_poses.shape == (N, d)
    assert errors.shape == (N,)


###################
# Test find_angle #
###################

def test_shapes_find_angles_grid():
    if torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'

    N, D, H, W = 150, 32, 32, 32
    reconstruction = torch.randn((D, H, W), device=device)
    patches = torch.randn((N, D, H, W), device=device)
    psf = torch.randn((D, H, W), device=device)

    best_poses, errors = find_angles_grid(reconstruction, patches, psf, precision=10)
    
    assert best_poses.shape == (N, 6)
    assert errors.shape == (N,)
