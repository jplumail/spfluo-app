# from spfluo.utils.loading import loadmat
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Tuple

import numpy as np
import pytest

import spfluo
from spfluo.refinement import (
    convolution_matching_poses_grid,
    convolution_matching_poses_refined,
    find_angles_grid,
    reconstruction_L2,
    refine,
)
from spfluo.tests.helpers import (
    assert_allclose,
    assert_volumes_aligned,
    ids,
    testing_libs,
)
from spfluo.utils.array import array_namespace, get_device, median
from spfluo.utils.transform import (
    distance_family_poses,
    get_transform_matrix,
    symmetrize_angles,
    symmetrize_poses,
)
from spfluo.utils.volume import (
    affine_transform,
)

if TYPE_CHECKING:
    from spfluo.utils.array import Array


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    yield
    if spfluo.has_torch_cuda():
        import torch

        torch.cuda.empty_cache()
    if spfluo.has_cupy():
        import cupy

        mempool = cupy.get_default_memory_pool()
        mempool.free_all_blocks()


@pytest.fixture(scope="module")
def generated_data_all_array(request, generated_data_all):
    xp, device = request.param
    volumes, poses, psf, groundtruth = tuple(
        [xp.asarray(x) for x in generated_data_all]
    )
    return (volumes, poses, psf, groundtruth), device


@pytest.fixture
def poses_with_noise(
    generated_data_all_array: Tuple["Array", ...],
):
    (_, poses, _, _), _ = generated_data_all_array
    xp = array_namespace(poses)
    device = get_device(poses)
    poses_noisy = xp.asarray(poses, copy=True)
    sigma_rot, sigma_trans = 20, 2
    np.random.seed(0)
    poses_noisy[:, :3] += (
        xp.asarray(np.random.randn(len(poses), 3), device=device) * sigma_rot
    )
    poses_noisy[:, 3:] += (
        xp.asarray(np.random.randn(len(poses), 3), device=device) * sigma_trans
    )
    return poses_noisy


##########################
# Test reconstruction_L2 #
##########################


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_shapes_reconstruction_L2(generated_data_all_array):
    (volumes, groundtruth_poses, psf, groundtruth), device = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0, device=get_device(volumes))
    recon = reconstruction_L2(volumes, psf, groundtruth_poses, lambda_, device=device)

    assert recon.shape == volumes.shape[-3:]


@pytest.mark.parametrize("minibatch", [1, 10, None])
@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_parallel_reconstruction_L2(minibatch, generated_data_all_array, save_result):
    (volumes, groundtruth_poses, psf, groundtruth), device = generated_data_all_array
    M = 3
    xp = array_namespace(volumes)
    lambda_ = xp.ones((M,))
    poses = groundtruth_poses + xp.asarray(
        np.random.randn(M, volumes.shape[0], 6) * 0.1
    )
    recon = reconstruction_L2(
        volumes, psf, poses, lambda_, batch=True, device=device, batch_size=minibatch
    )
    recon2 = xp.stack(
        [
            reconstruction_L2(volumes, psf, poses[i, ...], lambda_[i], device=device)
            for i in range(M)
        ]
    )

    save_result("reconstructions", recon2, metadata={"axes": "TCZYX"})
    save_result("reconstructions_paralled", recon, metadata={"axes": "TCZYX"})

    assert recon.shape == (M,) + volumes.shape[-3:]
    assert_allclose(recon, recon2, atol=1e-5)


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_multichannel_reconstruction_L2(
    generated_data_all_array, save_result: Callable[[str, np.ndarray], bool]
):
    # data preparation
    (volumes, poses, psf, groundtruth), compute_device = generated_data_all_array
    device = get_device(volumes)
    xp = array_namespace(volumes)
    volumes = xp.stack((volumes,) * 3, axis=1)
    psf = xp.stack((psf,) * 3, axis=0)
    dtype = volumes.dtype
    noise = xp.asarray(
        np.random.randn(*volumes.shape) * 0.2, device=device, dtype=dtype
    )
    volumes = volumes + noise
    lambda_ = xp.asarray(1.0)

    # reconstruction of 3-channel input should be the same as 3 mono-channel
    # input reconstrucions
    recon_multichannel = reconstruction_L2(
        volumes, psf, poses, lambda_, device=compute_device, multichannel=True
    )
    recon = xp.stack(
        [
            reconstruction_L2(
                volumes[:, i, ...],
                psf[i, ...],
                poses,
                lambda_,
                device=compute_device,
                multichannel=False,
            )
            for i in range(3)
        ]
    )

    save_result("reconstruction_multichannel", recon_multichannel)
    save_result("reconstruction", recon)

    assert recon_multichannel.shape == recon.shape
    assert_allclose(recon_multichannel, recon, atol=1e-4)


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_symmetry_reconstruction_L2(
    generated_data_all_array, save_result: Callable[[str, np.ndarray], bool]
):
    k = 9
    (volumes, poses, psf, groundtruth), device = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0)
    poses_sym = xp.stack((poses,) * k)  # useless symmetry
    recon_sym = reconstruction_L2(
        volumes, psf, poses_sym, lambda_, symmetry=True, device=device
    )
    recon = reconstruction_L2(
        xp.concat((volumes,) * k, axis=0),
        psf,
        xp.concat((poses,) * k, axis=0),
        lambda_,
        symmetry=False,
        device=device,
    )

    save_result("reconstruction_sym", recon_sym)
    save_result("reconstruction", recon)

    assert recon_sym.shape == volumes.shape[-3:]
    assert_allclose(recon_sym, recon, atol=1e-4)


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_symmetry_reconstruction_L2_2(
    generated_data_all_array: tuple["Array", ...],
    save_result: Callable[[str, np.ndarray], bool],
):
    """reconstruction_L2 of 1 particle with angles that have been symmetrized
    should get approximately the same result as a simple rotation
    """
    (volumes, groundtruth_poses, psf, groundtruth), device = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0)

    # select particle 0
    volume = volumes[0, ...][None, ...]
    pose = groundtruth_poses[0, ...][None, ...]

    # create symmetrical poses
    pose_sym = xp.stack((pose,) * 9)
    pose_sym[:, 0, :3] = symmetrize_angles(pose[0, :3], symmetry=9, degrees=True)

    # reconstruct
    recon_sym = reconstruction_L2(
        volume, psf, pose_sym, lambda_, symmetry=True, device=device
    )

    # compare with simple rotation
    rot = affine_transform(
        volume[0, ...],
        xp.asarray(
            get_transform_matrix(
                volume.shape[1:], pose[0, :3], pose[0, 3:], degrees=True
            ),
            dtype=volume.dtype,
        ),
        order=1,
    )

    # save and assert
    save_result("reconstruction_sym", recon_sym)
    save_result("simple_rot", rot)
    assert_volumes_aligned(recon_sym, rot)


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_reconstruction_L2_simple(
    generated_data_all_array: tuple["Array", ...],
    save_result: Callable[[str, np.ndarray], bool],
):
    """Do a reconstruction and compare if it's aligned with the groundtruth"""
    (volumes, groundtruth_poses, psf, groundtruth), device = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0)
    reconstruction = reconstruction_L2(
        volumes, psf, groundtruth_poses, lambda_, device=device
    )

    save_result("reconstruction", reconstruction)
    save_result("groundtruth", groundtruth)

    assert_volumes_aligned(reconstruction, groundtruth, atol=1)


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_reconstruction_L2_symmetry(
    generated_data_all_array: tuple["Array", ...],
    save_result: Callable[[str, np.ndarray], bool],
):
    """Do a reconstruction with symmetry and compare if it's aligned with groundtruth"""
    (volumes, groundtruth_poses, psf, groundtruth), device = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0)
    euler_angles_sym = symmetrize_angles(
        groundtruth_poses[:, :3], symmetry=9, degrees=True
    )
    gt_poses_sym = xp.concat(
        (euler_angles_sym, xp.zeros_like(euler_angles_sym)), axis=2
    )
    gt_poses_sym[:, :, 3:] = groundtruth_poses[:, None, 3:]  # shape (N, k, 6)
    gt_poses_sym = xp.permute_dims(gt_poses_sym, (1, 0, 2))  # shape (k, N, 6)

    reconstruction = reconstruction_L2(
        volumes, psf, gt_poses_sym, lambda_, symmetry=True, device=device
    )

    save_result("reconstruction", reconstruction)
    save_result("groundtruth", groundtruth)

    assert_volumes_aligned(reconstruction, groundtruth, atol=1)


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_reconstruction_L2_symmetry_1vol_iso(
    generated_data_all_array: tuple["Array", ...],
    save_result: Callable[[str, np.ndarray], bool],
):
    """Do a reconstruction with 1 volume in 2 ways:
        - 1 volume, 9 poses, reconstruction_L2 with symmetry=True
        - 9x 1 volume, 9 poses, reconstruction_L2 with symmetry=False

    The results must be the same.
    """
    (volumes, groundtruth_poses, psf, groundtruth), device = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0)
    euler_angles_sym = symmetrize_angles(
        groundtruth_poses[:, :3], symmetry=9, degrees=True
    )
    gt_poses_sym = xp.concat(
        (euler_angles_sym, xp.zeros_like(euler_angles_sym)), axis=2
    )
    gt_poses_sym[:, :, 3:] = groundtruth_poses[:, None, 3:]  # shape (N, k, 6)
    gt_poses_sym = xp.permute_dims(gt_poses_sym, (1, 0, 2))  # shape (k, N, 6)

    reconstruction_sym = reconstruction_L2(
        volumes[:1, ...],
        psf,
        gt_poses_sym[:, :1, :],
        lambda_,
        symmetry=True,
        device=device,
    )
    volume0_repeated = xp.stack((volumes[0, ...],) * 9)
    reconstruction = reconstruction_L2(
        volume0_repeated,
        psf,
        gt_poses_sym[:, 0, :],
        lambda_,
        symmetry=False,
        device=device,
    )

    save_result("reconstruction_sym=True", reconstruction_sym)
    save_result("reconstruction_sym=False", reconstruction)

    assert_allclose(reconstruction_sym, reconstruction, rtol=0.01, atol=1e-5)


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_reconstruction_L2_symmetry_Nvol_iso(
    generated_data_all_array: tuple["Array", ...],
    save_result: Callable[[str, np.ndarray], bool],
):
    """Do a reconstruction with N volumes in 2 ways:
        - N volumes, Nx9 poses, reconstruction_L2 with symmetry=True
        - 9xN volumes, Nx9 poses, reconstruction_L2 with symmetry=False

    The results must be the same.
    """
    (volumes, groundtruth_poses, psf, groundtruth), device = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0)
    euler_angles_sym = symmetrize_angles(
        groundtruth_poses[:, :3], symmetry=9, degrees=True
    )
    gt_poses_sym = xp.concat(
        (euler_angles_sym, xp.zeros_like(euler_angles_sym)), axis=2
    )
    gt_poses_sym[:, :, 3:] = groundtruth_poses[:, None, 3:]  # shape (N, k, 6)
    gt_poses_sym = xp.permute_dims(gt_poses_sym, (1, 0, 2))  # shape (k, N, 6)

    reconstruction_sym = reconstruction_L2(
        volumes, psf, gt_poses_sym, lambda_, symmetry=True, device=device
    )
    volumes_repeated = xp.concat(
        [xp.stack((volumes[i, ...],) * 9) for i in range(volumes.shape[0])]
    )
    reconstruction = reconstruction_L2(
        volumes_repeated,
        psf,
        xp.reshape(xp.permute_dims(gt_poses_sym, (1, 0, 2)), (-1, 6)),
        lambda_,
        symmetry=False,
        device=device,
    )

    save_result("reconstruction_sym=True", reconstruction_sym)
    save_result("reconstruction_sym=False", reconstruction)

    assert_allclose(reconstruction_sym, reconstruction, rtol=0.01, atol=1e-6)


#############################
# Test convolution_matching #
#############################

gpu_libs, gpu_ids = zip(
    *filter(
        lambda x: x[0][1] == "cuda" or "cupy" in x[0][0].__name__,
        zip(testing_libs, ids),
    )
)


# This test requires ~12GB VRAM
@pytest.mark.slow
@pytest.mark.parametrize(
    "generated_data_all_array", gpu_libs, indirect=True, ids=gpu_ids
)
def test_memory_convolution_matching_poses_grid(generated_data_all_array):
    """Test if a out of memory error occurs"""
    (volumes, groundtruth_poses, psf, groundtruth), device = generated_data_all_array
    xp = array_namespace(volumes)
    # 10 volumes of size 50^3, float32 -> 5 Mo
    # M = 10_000 -> 50 Go to transfer to GPU
    # We split the computation in batch_size
    for M in [1, 10, 100, 1_000, 10_000]:
        potential_poses = xp.asarray(np.random.randn(M, 6), dtype=volumes.dtype)

        best_poses, errors = convolution_matching_poses_grid(
            groundtruth[None],
            volumes[:, None],
            psf[None],
            potential_poses,
            device=device,
            batch_size=256,
        )

        assert best_poses.shape == (volumes.shape[0], 6)
        assert errors.shape == (volumes.shape[0],)


@pytest.mark.parametrize("xp,device", gpu_libs, ids=gpu_ids)
def test_shapes_convolution_matching_poses_grid(xp, device):
    M, d = 5, 6
    N, C, D, H, W = 10, 2, 32, 32, 32
    reference = xp.asarray(np.random.randn(C, D, H, W), device=device)
    volumes = xp.asarray(np.random.randn(N, C, D, H, W), device=device)
    psf = xp.asarray(np.random.randn(C, D, H, W), device=device)
    potential_poses = xp.asarray(np.random.randn(M, d), device=device)

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


@pytest.mark.parametrize("xp,device", gpu_libs, ids=gpu_ids)
def test_shapes_convolution_matching_poses_refined(xp, device):
    M, d = 5, 6
    N, C, D, H, W = 10, 2, 32, 32, 32
    reference = xp.asarray(np.random.randn(C, D, H, W), device=device)
    volumes = xp.asarray(np.random.randn(N, C, D, H, W), device=device)
    psf = xp.asarray(np.random.randn(C, D, H, W), device=device)
    potential_poses = xp.asarray(np.random.randn(N, M, d), device=device)

    best_poses, errors = convolution_matching_poses_refined(
        reference, volumes, psf, potential_poses, device=device, batch_size=2048
    )

    assert best_poses.shape == (N, d)
    assert errors.shape == (N,)


###################
# Test find_angle #
###################


@pytest.mark.slow
@pytest.mark.parametrize("xp,device", gpu_libs, ids=gpu_ids)
def test_shapes_find_angles_grid(xp, device):
    N, C, D, H, W = 15, 2, 32, 32, 32
    reconstruction = xp.asarray(np.random.randn(C, D, H, W), device=device)
    patches = xp.asarray(np.random.randn(N, C, D, H, W), device=device)
    psf = xp.asarray(np.random.randn(C, D, H, W), device=device)

    best_poses, errors = find_angles_grid(
        xp, reconstruction, patches, psf, precision=10
    )

    assert best_poses.shape == (N, 6)
    assert errors.shape == (N,)


###############
# Test refine #
###############


# batch_size is adjusted for ~12GB VRAM
@pytest.mark.slow
@pytest.mark.parametrize("xp,device", gpu_libs, ids=gpu_ids)
def test_refine_shapes(xp, device):
    N, C, D, H, W = 15, 2, 32, 32, 32
    patches = xp.asarray(np.random.randn(N, C, D, H, W))
    psf = xp.asarray(np.random.randn(C, D, H, W))
    guessed_poses = xp.asarray(np.random.randn(N, 6))

    S = 2
    steps = [(S * S, S), S * S * S]
    ranges = [0, 40]
    recon, poses = refine(
        patches, psf, guessed_poses, steps, ranges, device=device, batch_size=256
    )

    assert recon.shape == patches[0].shape
    assert poses.shape == guessed_poses.shape


@pytest.mark.slow
@pytest.mark.parametrize("xp,device", gpu_libs, ids=gpu_ids)
def test_refine_initial_vol_shapes(xp, device):
    N, C, D, H, W = 15, 2, 32, 32, 32
    patches = xp.asarray(np.random.randn(N, C, D, H, W))
    psf = xp.asarray(np.random.randn(C, D, H, W))
    guessed_poses = xp.asarray(np.random.randn(N, 6))
    initial_volume = xp.asarray(np.random.randn(C, D, H, W))

    S = 2
    steps = [(S * S, S), S * S * S]
    ranges = [0, 40]
    recon, poses = refine(
        patches,
        psf,
        guessed_poses,
        steps,
        ranges,
        device=device,
        batch_size=256,
        initial_volume=initial_volume,
    )

    assert recon.shape == patches[0].shape
    assert poses.shape == guessed_poses.shape


@pytest.mark.parametrize(
    "generated_data_all_array", gpu_libs, indirect=True, ids=gpu_ids
)
def test_refine_easy(
    generated_data_all_array: tuple["Array", ...],
    poses_with_noise: "Array",
    save_result: Callable[[str, np.ndarray], bool],
):
    poses = poses_with_noise

    # data preparation
    (volumes, groundtruth_poses, psf, groundtruth), compute_device = (
        generated_data_all_array
    )
    xp = array_namespace(volumes)

    lambda_ = xp.asarray(1.0, device=get_device(volumes))
    poses_sym = xp.permute_dims(symmetrize_poses(poses, 9), (1, 0, 2))
    initial_reconstruction = reconstruction_L2(
        volumes,
        psf,
        poses_sym,
        lambda_,
        symmetry=True,
        device=compute_device,
        batch_size=1,
    )

    S = 5
    A = 5 * 2
    steps = [(A**2, 5)] + [S] * 7  # 7.25° axis precision; 4° sym precision
    ranges = [
        0,
    ] + [10, 5, 5, 2, 2, 1, 1]
    reconstruction, best_poses = refine(
        volumes[:, None],
        psf[None],
        poses,
        steps,
        ranges,
        symmetry=9,
        lambda_=1e-2,
        device=compute_device,
        batch_size=256,
    )
    reconstruction = reconstruction[0]

    rot_dist_deg1, trans_dist_pix1 = distance_family_poses(
        best_poses, groundtruth_poses, symmetry=9
    )
    rot_dist_deg2, trans_dist_pix2 = distance_family_poses(
        poses, groundtruth_poses, symmetry=9
    )

    save_result("initial_reconstruction", initial_reconstruction)
    save_result("final_reconstruction", reconstruction)

    assert median(rot_dist_deg1) < median(rot_dist_deg2) and median(
        trans_dist_pix1
    ) < median(trans_dist_pix2)


@pytest.mark.parametrize(
    "generated_data_all_array", gpu_libs, indirect=True, ids=gpu_ids
)
def test_refine_easy_multichannel(
    generated_data_all_array: tuple["Array", ...],
    poses_with_noise: "Array",
    save_result: Callable[[str, np.ndarray], bool],
):
    poses = poses_with_noise

    # data preparation
    (volumes, groundtruth_poses, psf, groundtruth), compute_device = (
        generated_data_all_array
    )
    device = get_device(volumes)
    xp = array_namespace(volumes)
    volumes = xp.stack((volumes,) * 3, axis=1)
    psf = xp.stack((psf,) * 3, axis=0)
    dtype = volumes.dtype
    noise = xp.asarray(
        np.random.randn(*volumes.shape) * 0.2, device=device, dtype=dtype
    )
    volumes = volumes + noise
    lambda_ = xp.asarray(1.0)

    lambda_ = xp.asarray(1.0, device=get_device(volumes))
    poses_sym = xp.permute_dims(symmetrize_poses(poses, 9), (1, 0, 2))
    initial_reconstruction = reconstruction_L2(
        volumes,
        psf,
        poses_sym,
        lambda_,
        symmetry=True,
        device=compute_device,
        batch_size=1,
        multichannel=True,
    )

    S = 5
    A = 5 * 2
    steps = [(A**2, 5)] + [S] * 7  # 7.25° axis precision; 4° sym precision
    ranges = [
        0,
    ] + [10, 5, 5, 2, 2, 1, 1]
    reconstruction, best_poses = refine(
        volumes,
        psf,
        poses,
        steps,
        ranges,
        symmetry=9,
        lambda_=1e-2,
        device=compute_device,
        batch_size=256,
    )

    rot_dist_deg1, trans_dist_pix1 = distance_family_poses(
        best_poses, groundtruth_poses, symmetry=9
    )
    rot_dist_deg2, trans_dist_pix2 = distance_family_poses(
        poses, groundtruth_poses, symmetry=9
    )

    save_result("initial_reconstruction", initial_reconstruction)
    save_result("final_reconstruction", reconstruction)

    assert median(rot_dist_deg1) < median(rot_dist_deg2) and median(
        trans_dist_pix1
    ) < median(trans_dist_pix2)
