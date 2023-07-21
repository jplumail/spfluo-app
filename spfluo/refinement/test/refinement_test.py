# from spfluo.utils.loading import loadmat

from typing import Tuple

import torch

from spfluo.refinement import (
    convolution_matching_poses_grid,
    convolution_matching_poses_refined,
    find_angles_grid,
    reconstruction_L2,
    refine,
)
from spfluo.utils.transform import distance_family_poses
from spfluo.utils.volume import are_volumes_aligned

##########################
# Test reconstruction_L2 #
##########################


def test_shapes_reconstruction_L2():
    N, D, H, W = 10, 32, 32, 32
    volumes = torch.randn((N, D, H, W))
    psf = torch.randn((D, H, W))
    poses = torch.randn((N, 6))
    lambda_ = torch.tensor(1.0)
    recon, den = reconstruction_L2(volumes, psf, poses, lambda_)

    assert recon.shape == (D, H, W)
    assert den.shape == (D, H, W)


def test_parallel_reconstruction_L2():
    M = 5
    N, D, H, W = 10, 32, 32, 32
    volumes = torch.randn((N, D, H, W))
    psf = torch.randn((D, H, W))
    poses = torch.randn((M, N, 6))
    lambda_ = torch.randn((M,))
    recon, _ = reconstruction_L2(volumes, psf, poses, lambda_)
    recon2 = torch.stack(
        [reconstruction_L2(volumes, psf, poses[i], lambda_[i])[0] for i in range(M)]
    )

    assert recon.shape == (M, D, H, W)
    assert torch.isclose(recon, recon2).all()


def test_reconstruction_L2_simple(generated_data_pytorch: Tuple[torch.Tensor, ...]):
    volumes, groundtruth_poses, psf, groundtruth = generated_data_pytorch
    lbda = volumes.new_tensor(1e-5)
    reconstruction, _ = reconstruction_L2(volumes, psf, groundtruth_poses, lbda)

    assert are_volumes_aligned(
        reconstruction.cpu().numpy(), groundtruth.cpu().numpy(), atol=1
    )


#############################
# Test convolution_matching #
#############################


def test_memory_convolution_matching_poses_grid():
    device = "cuda"
    D = 32
    for N in [1, 10]:
        N = int(N)
        M = int(100 / N**0.5)
        reference = torch.randn((D, D, D), device=device)
        volumes = torch.randn((N, D, D, D), device=device)
        psf = torch.randn((D, D, D), device=device)
        potential_poses = torch.randn((M, 6), device=device)

        best_poses, errors = convolution_matching_poses_grid(
            reference, volumes, psf, potential_poses
        )

        assert best_poses.shape == (N, 6)
        assert errors.shape == (N,)


def test_shapes_convolution_matching_poses_grid():
    M, d = 5, 6
    N, D, H, W = 100, 32, 32, 32
    reference = torch.randn((D, H, W))
    volumes = torch.randn((N, D, H, W))
    psf = torch.randn((D, H, W))
    potential_poses = torch.randn((M, d))

    best_poses, errors = convolution_matching_poses_grid(
        reference, volumes, psf, potential_poses
    )

    assert best_poses.shape == (N, d)
    assert errors.shape == (N,)


# TODO faire les tests matlab
# def test_matlab_convolution_matching_poses_refined():
#     def as_tensor(x):
#         return torch.as_tensor(x, dtype=torch.float64, device="cuda")
#
#     # Load Matlab data
#     data_path = \
#       os.path.join(os.path.dirname(__file__), "data", "convolution_matching")
#     potential_poses_ = loadmat(os.path.join(data_path, "bigListPoses.mat"))[
#         "bigListPoses"
#     ]
#     volumes = np.stack(
#         loadmat(os.path.join(data_path, "inVols.mat"))["inVols"][:, 0]
#     ).transpose(0, 3, 2, 1)
#     best_poses_matlab = loadmat(os.path.join(data_path, "posesNew.mat"))["posesNew"][
#         :, [0, 1, 2, 5, 3, 4]
#     ]
#     best_poses_matlab[:, 3:] *= -1
#     psf = loadmat(os.path.join(data_path, "psf.mat"))["psf"].transpose(2, 1, 0)
#     reference = loadmat(os.path.join(data_path, "recon.mat"))["recon1"].transpose(
#         2, 1, 0
#     )
#
#     potential_poses_, volumes, best_poses_matlab, psf, reference = map(
#         as_tensor, [potential_poses_, volumes, best_poses_matlab, psf, reference]
#     )
#
#     N, M, _ = potential_poses_.shape
#     potential_poses = as_tensor(torch.zeros((N, M, 6)))
#     potential_poses[:, :, :3] = potential_poses_
#
#     best_poses, _ = convolution_matching_poses_refined(
#         reference, volumes, psf, potential_poses
#     )
#
#     eps = 1e-2
#     assert ((best_poses - best_poses_matlab) < eps).all()


def test_shapes_convolution_matching_poses_refined():
    M, d = 5, 6
    N, D, H, W = 10, 32, 32, 32
    reference = torch.randn((D, H, W))
    volumes = torch.randn((N, D, H, W))
    psf = torch.randn((D, H, W))
    potential_poses = torch.randn((N, M, d))

    best_poses, errors = convolution_matching_poses_refined(
        reference, volumes, psf, potential_poses
    )

    assert best_poses.shape == (N, d)
    assert errors.shape == (N,)


###################
# Test find_angle #
###################


def test_shapes_find_angles_grid():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    N, D, H, W = 15, 32, 32, 32
    reconstruction = torch.randn((D, H, W), device=device)
    patches = torch.randn((N, D, H, W), device=device)
    psf = torch.randn((D, H, W), device=device)

    best_poses, errors = find_angles_grid(reconstruction, patches, psf, precision=10)

    assert best_poses.shape == (N, 6)
    assert errors.shape == (N,)


###############
# Test refine #
###############


def test_refine_shapes():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    N, D, H, W = 15, 32, 32, 32
    patches = torch.randn((N, D, H, W), device=device)
    psf = torch.randn((D, H, W), device=device)
    guessed_poses = torch.randn((N, 6), device=device)

    S = 2
    steps = [(S * S, S), S * S * S]
    ranges = [0, 40]
    recon, poses = refine(patches, psf, guessed_poses, steps, ranges)

    assert recon.shape == patches[0].shape
    assert poses.shape == guessed_poses.shape


def test_refine_easy(generated_data_pytorch, poses_with_noise_pytorch):
    poses = poses_with_noise_pytorch
    volumes, groundtruth_poses, psf, groundtruth = generated_data_pytorch

    S = 10
    A = 14 * 2
    steps = [(A**2, 50)] + [S] * 7  # 7.25° axis precision; 0.8° sym precision
    ranges = [
        0,
    ] + [10, 5, 5, 2, 2, 1, 1]
    reconstruction, best_poses = refine(
        volumes, psf, poses, steps, ranges, symmetry=9, lambda_=1e-2
    )

    rot_dist_deg1, trans_dist_pix1 = distance_family_poses(
        best_poses, groundtruth_poses, symmetry=9
    )
    rot_dist_deg2, trans_dist_pix2 = distance_family_poses(
        poses, groundtruth_poses, symmetry=9
    )

    assert rot_dist_deg1.sum() < rot_dist_deg2.sum()
    assert trans_dist_pix1.sum() < trans_dist_pix2.sum()
