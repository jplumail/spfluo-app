# from spfluo.utils.loading import loadmat

from typing import Callable, Tuple

import numpy as np
import pytest

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
from spfluo.utils.array import Array, array_namespace
from spfluo.utils.transform import (
    distance_family_poses,
    get_transform_matrix,
    symmetrize_angles,
    symmetrize_poses,
)
from spfluo.utils.volume import (
    affine_transform,
)

testing_libs, ids = zip(
    *filter(lambda x: "torch" in x[0][0].__name__, zip(testing_libs, ids))
)


@pytest.fixture(scope="module")
def generated_data_all_array(request, generated_data_all):
    xp, device = request.param
    return tuple([xp.asarray(x, device=device) for x in generated_data_all])


@pytest.fixture
def poses_with_noise(
    generated_data_all_array: Tuple[Array, ...],
):
    _, poses, _, _ = generated_data_all_array
    xp = array_namespace(poses)
    device = xp.device(poses)
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
    volumes, groundtruth_poses, psf, groundtruth = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0, device=xp.device(volumes))
    recon = reconstruction_L2(volumes, psf, groundtruth_poses, lambda_)

    assert recon.shape == volumes.shape[-3:]


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_parallel_reconstruction_L2(generated_data_all_array, save_result):
    volumes, groundtruth_poses, psf, groundtruth = generated_data_all_array
    M = 3
    xp = array_namespace(volumes)
    lambda_ = xp.ones((M,), device=xp.device(volumes))
    poses = groundtruth_poses + xp.asarray(
        np.random.randn(M, volumes.shape[0], 6) * 0.1, device=xp.device(volumes)
    )
    recon = reconstruction_L2(volumes, psf, poses, lambda_, batch=True)
    recon2 = xp.stack(
        [reconstruction_L2(volumes, psf, poses[i], lambda_[i]) for i in range(M)]
    )

    save_result("reconstructions", recon2, metadata={"axes": "TZYX"})
    save_result("reconstructions_paralled", recon, metadata={"axes": "TZYX"})

    assert recon.shape == (M,) + volumes.shape[-3:]
    assert_allclose(recon, recon2, atol=1e-5)


@pytest.mark.xfail(run=False)
@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_symmetry_reconstruction_L2(
    generated_data_all_array, save_result: Callable[[str, np.ndarray], bool]
):
    k = 9
    volumes, poses, psf, groundtruth = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0, device=xp.device(volumes))
    poses = xp.stack((poses,) * k)  # useless symmetry
    recon_sym = reconstruction_L2(volumes, psf, poses, lambda_, symmetry=True)
    recon = reconstruction_L2(volumes, psf, poses[0], lambda_, symmetry=False)

    save_result("reconstruction_sym", recon_sym)
    save_result("reconstruction", recon)

    assert recon_sym.shape == volumes.shape[-3:]
    assert_allclose(recon_sym, recon, atol=1e-4)


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_symmetry_reconstruction_L2_2(
    generated_data_all_array: tuple[Array, ...],
    save_result: Callable[[str, np.ndarray], bool],
):
    """reconstruction_L2 of 1 particle with angles that have been symmetrized
    should get approximately the same result as a simple rotation
    """
    volumes, groundtruth_poses, psf, groundtruth = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0, device=xp.device(volumes))

    # select particle 0
    volume = volumes[0][None]
    pose = groundtruth_poses[0][None]

    # create symmetrical poses
    pose_sym = xp.stack((pose,) * 9)
    pose_sym[:, 0, :3] = symmetrize_angles(pose[0, :3], symmetry=9, degrees=True)

    # reconstruct
    recon_sym = reconstruction_L2(volume, psf, pose_sym, lambda_, symmetry=True)

    # compare with simple rotation
    rot = affine_transform(
        volume[0],
        xp.asarray(
            get_transform_matrix(
                volume[0].shape, pose[0, :3], pose[0, 3:], degrees=True
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
    generated_data_all_array: tuple[Array, ...],
    save_result: Callable[[str, np.ndarray], bool],
):
    """Do a reconstruction and compare if it's aligned with the groundtruth"""
    volumes, groundtruth_poses, psf, groundtruth = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0, device=xp.device(volumes))
    reconstruction = reconstruction_L2(volumes, psf, groundtruth_poses, lambda_)

    save_result("reconstruction", reconstruction)
    save_result("groundtruth", groundtruth)

    assert_volumes_aligned(reconstruction, groundtruth, atol=1)


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_reconstruction_L2_symmetry(
    generated_data_all_array: tuple[Array, ...],
    save_result: Callable[[str, np.ndarray], bool],
):
    """Do a reconstruction with symmetry and compare if it's aligned with groundtruth"""
    volumes, groundtruth_poses, psf, groundtruth = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0, device=xp.device(volumes))
    euler_angles_sym = symmetrize_angles(
        groundtruth_poses[:, :3], symmetry=9, degrees=True
    )
    gt_poses_sym = xp.concat(
        (euler_angles_sym, xp.zeros_like(euler_angles_sym)), axis=2
    )
    gt_poses_sym[:, :, 3:] = groundtruth_poses[:, None, 3:]  # shape (N, k, 6)
    gt_poses_sym = xp.permute_dims(gt_poses_sym, (1, 0, 2))  # shape (k, N, 6)

    reconstruction = reconstruction_L2(
        volumes, psf, gt_poses_sym, lambda_, symmetry=True
    )

    save_result("reconstruction", reconstruction)
    save_result("groundtruth", groundtruth)

    assert_volumes_aligned(reconstruction, groundtruth, atol=1)


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_reconstruction_L2_symmetry_1vol_iso(
    generated_data_all_array: tuple[Array, ...],
    save_result: Callable[[str, np.ndarray], bool],
):
    """Do a reconstruction with 1 volume in 2 ways:
        - 1 volume, 9 poses, reconstruction_L2 with symmetry=True
        - 9x 1 volume, 9 poses, reconstruction_L2 with symmetry=False

    The results must be the same.
    """
    volumes, groundtruth_poses, psf, groundtruth = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0, device=xp.device(volumes))
    euler_angles_sym = symmetrize_angles(
        groundtruth_poses[:, :3], symmetry=9, degrees=True
    )
    gt_poses_sym = xp.cat((euler_angles_sym, xp.zeros_like(euler_angles_sym)), 2)
    gt_poses_sym[:, :, 3:] = groundtruth_poses[:, None, 3:]  # shape (N, k, 6)
    gt_poses_sym = xp.permute_dims(gt_poses_sym, (1, 0, 2))  # shape (k, N, 6)

    reconstruction_sym = reconstruction_L2(
        volumes[:1], psf, gt_poses_sym[:, :1, :], lambda_, symmetry=True
    )
    volume0_repeated = xp.stack((volumes[0],) * 9)
    reconstruction = reconstruction_L2(
        volume0_repeated, psf, gt_poses_sym[:, 0], lambda_, symmetry=False
    )

    save_result("reconstruction_sym=True", reconstruction_sym)
    save_result("reconstruction_sym=False", reconstruction)

    assert_allclose(reconstruction_sym, reconstruction, rtol=0.01, atol=1e-5)


@pytest.mark.parametrize(
    "generated_data_all_array", testing_libs, indirect=True, ids=ids
)
def test_reconstruction_L2_symmetry_Nvol_iso(
    generated_data_all_array: tuple[Array, ...],
    save_result: Callable[[str, np.ndarray], bool],
):
    """Do a reconstruction with N volumes in 2 ways:
        - N volumes, Nx9 poses, reconstruction_L2 with symmetry=True
        - 9xN volumes, Nx9 poses, reconstruction_L2 with symmetry=False

    The results must be the same.
    """
    volumes, groundtruth_poses, psf, groundtruth = generated_data_all_array
    xp = array_namespace(volumes)
    lambda_ = xp.asarray(1.0, device=xp.device(volumes))
    euler_angles_sym = symmetrize_angles(
        groundtruth_poses[:, :3], symmetry=9, degrees=True
    )
    gt_poses_sym = xp.concat(
        (euler_angles_sym, xp.zeros_like(euler_angles_sym)), axis=2
    )
    gt_poses_sym[:, :, 3:] = groundtruth_poses[:, None, 3:]  # shape (N, k, 6)
    gt_poses_sym = xp.permute_dims(gt_poses_sym, (1, 0, 2))  # shape (k, N, 6)

    reconstruction_sym = reconstruction_L2(
        volumes, psf, gt_poses_sym, lambda_, symmetry=True
    )
    volumes_repeated = xp.concat(
        [xp.stack((volumes[i],) * 9) for i in range(volumes.shape[0])]
    )
    reconstruction = reconstruction_L2(
        volumes_repeated,
        psf,
        xp.permute_dims(gt_poses_sym, (1, 0, 2)).reshape(-1, 6),
        lambda_,
        symmetry=False,
    )

    save_result("reconstruction_sym=True", reconstruction_sym)
    save_result("reconstruction_sym=False", reconstruction)

    assert_allclose(reconstruction_sym, reconstruction, rtol=0.01, atol=1e-6)


#############################
# Test convolution_matching #
#############################

gpu_libs, gpu_ids = zip(
    *filter(
        lambda x: "cuda" in x[0][1] or "cupy" in x[0][0].__name__,
        zip(testing_libs, ids),
    )
)


@pytest.mark.parametrize(
    "generated_data_all_array", gpu_libs, indirect=True, ids=gpu_ids
)
def test_memory_convolution_matching_poses_grid(generated_data_all_array):
    """Test if a out of memory error occurs"""
    volumes, groundtruth_poses, psf, groundtruth = generated_data_all_array
    xp = array_namespace(volumes)
    # 10 volumes of size 50^3, float32 -> 5 Mo
    # M = 10_000 -> 50 Go to transfer to GPU
    for M in [1, 10, 100, 1_000, 10_000]:
        potential_poses = xp.asarray(
            np.random.randn(M, 6), device=xp.device(volumes), dtype=volumes.dtype
        )

        best_poses, errors = convolution_matching_poses_grid(
            groundtruth, volumes, psf, potential_poses
        )

        assert best_poses.shape == (volumes.shape[0], 6)
        assert errors.shape == (volumes.shape[0],)


@pytest.mark.parametrize("xp,device", testing_libs, ids=ids)
def test_shapes_convolution_matching_poses_grid(xp, device):
    M, d = 5, 6
    N, D, H, W = 10, 32, 32, 32
    reference = xp.asarray(np.random.randn(D, H, W), device=device)
    volumes = xp.asarray(np.random.randn(N, D, H, W), device=device)
    psf = xp.asarray(np.random.randn(D, H, W), device=device)
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


@pytest.mark.parametrize("xp,device", testing_libs, ids=ids)
def test_shapes_convolution_matching_poses_refined(xp, device):
    M, d = 5, 6
    N, D, H, W = 10, 32, 32, 32
    reference = xp.asarray(np.random.randn(D, H, W), device=device)
    volumes = xp.asarray(np.random.randn(N, D, H, W), device=device)
    psf = xp.asarray(np.random.randn(D, H, W), device=device)
    potential_poses = xp.asarray(np.random.randn(N, M, d), device=device)

    best_poses, errors = convolution_matching_poses_refined(
        reference, volumes, psf, potential_poses
    )

    assert best_poses.shape == (N, d)
    assert errors.shape == (N,)


###################
# Test find_angle #
###################


@pytest.mark.parametrize("xp,device", testing_libs, ids=ids)
def test_shapes_find_angles_grid(xp, device):
    N, D, H, W = 15, 32, 32, 32
    reconstruction = xp.asarray(np.random.randn(D, H, W), device=device)
    patches = xp.asarray(np.random.randn(N, D, H, W), device=device)
    psf = xp.asarray(np.random.randn(D, H, W), device=device)

    best_poses, errors = find_angles_grid(reconstruction, patches, psf, precision=10)

    assert best_poses.shape == (N, 6)
    assert errors.shape == (N,)


###############
# Test refine #
###############


@pytest.mark.parametrize("xp,device", testing_libs, ids=ids)
def test_refine_shapes(xp, device):
    N, D, H, W = 15, 32, 32, 32
    patches = xp.asarray(np.random.randn(N, D, H, W), device=device)
    psf = xp.asarray(np.random.randn(D, H, W), device=device)
    guessed_poses = xp.asarray(np.random.randn(N, 6), device=device)

    S = 2
    steps = [(S * S, S), S * S * S]
    ranges = [0, 40]
    recon, poses = refine(patches, psf, guessed_poses, steps, ranges)

    assert recon.shape == patches[0].shape
    assert poses.shape == guessed_poses.shape


# @pytest.mark.skipif(device == "cpu", reason="Too long if done on CPU.")
@pytest.mark.xfail(
    reason="distance_family_poses doesn't work as expected, "
    "the final reconstruction is better than the initial one.",
    run=True,
)
@pytest.mark.parametrize(
    "generated_data_all_array", gpu_libs, indirect=True, ids=gpu_ids
)
def test_refine_easy(
    generated_data_all_array: tuple[Array, ...],
    poses_with_noise: Array,
    save_result: Callable[[str, np.ndarray], bool],
):
    poses = poses_with_noise
    volumes, groundtruth_poses, psf, groundtruth = generated_data_all_array
    xp = array_namespace(volumes)

    lambda_ = xp.asarray(1.0, device=xp.device(volumes))
    initial_reconstruction = reconstruction_L2(
        volumes,
        psf,
        xp.permute_dims(symmetrize_poses(poses, 9), (1, 0, 2)),
        lambda_,
        symmetry=True,
    )

    S = 5
    A = 5 * 2
    steps = [(A**2, 5)] + [S] * 7  # 7.25° axis precision; 4° sym precision
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

    save_result("initial_reconstruction", initial_reconstruction)
    save_result("final_reconstruction", reconstruction)

    assert xp.mean(rot_dist_deg1) < xp.mean(rot_dist_deg2) and xp.mean(
        trans_dist_pix1
    ) < xp.mean(trans_dist_pix2)
