import numpy as np
from scipy.spatial.transform import Rotation as R

from spfluo import data
from spfluo.utils.transform import (
    distance_family_poses,
    distance_poses,
    get_transform_matrix,
    symmetrize_poses,
)
from spfluo.utils.volume import affine_transform, are_volumes_aligned


def test_get_transform_matrix_simple():
    H = get_transform_matrix(
        (30, 30, 30),
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        convention="ZXZ",
        degrees=False,
    )
    expected_result = np.array(
        [
            [-0.48547846, -0.42291857, 0.7651474, 20.57711966],
            [-0.8647801, 0.10384657, -0.4912955, 37.65732099],
            [0.12832006, -0.90019763, -0.41614684, 37.72635389],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.isclose(H, expected_result).all()


def test_get_transform_matrix_batch():
    output_shape = (30, 30, 30)
    N = 10
    rot = np.random.randn(N, 3)
    trans = np.random.randn(N, 3)

    matrices = get_transform_matrix(output_shape, rot, trans)
    for i in range(N):
        assert np.isclose(
            matrices[i], get_transform_matrix(output_shape, rot[i], trans[i])
        ).all()


def test_distance_poses():
    p1 = np.asarray([90, 90, 0, 1, 0, 0], dtype=float)
    p2 = np.asarray([-90, 90, 0, 0, 1, 0], dtype=float)
    angle, t = distance_poses(p1, p2)

    assert np.isclose(angle, 180.0)
    assert np.isclose(t, 2.0**0.5)


def test_distance_family_poses():
    p1 = np.asarray(
        [[70, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [90, 90, 0, 1, 0, 0]], dtype=float
    )
    p2 = np.asarray(
        [[70, 0, 0, 0, 0, 0], [90, 0, 0, 0, 0, 0], [90, 90, 0, 0, 1, 0]], dtype=float
    )
    angle, t = distance_family_poses(p1, p2)
    assert np.isclose(angle, [0, 90, 0], atol=1e-5).all()
    assert np.isclose(t, [0, 0, 2.0**0.5]).all()


def test_distance_family_poses_sym():
    symmetry = 9
    error = 0.2 * (360 / symmetry)
    p1 = np.asarray(
        [
            [70, 0, 0, 0, 0, 0],
            [5 * 360 / symmetry + error, 0, 0, 0, 0, 0],
            [90, 90, 0, 1, 0, 0],
        ],
        dtype=float,
    )
    p2 = np.asarray(
        [[70, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [90, 90, 0, 0, 1, 0]], dtype=float
    )
    angle, t = distance_family_poses(p1, p2, symmetry=9)
    assert np.isclose(angle, [0, error, 0], atol=1e-5).all()
    assert np.isclose(t, [0, 0, 2.0**0.5]).all()


def test_distance_family_poses_rotoffset():
    p2 = np.asarray(
        [[70, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [90, 90, 0, 0, 0, 0]], dtype=float
    )
    R_p2 = R.from_euler("XZX", p2[:, :3], degrees=True)
    R0 = R.random(random_state=0)
    R_p1 = R_p2 * R0
    p1 = np.zeros_like(p2)
    p1[:, :3] = R_p1.as_euler("XZX", degrees=True)
    angle, t = distance_family_poses(p1, p2)
    assert np.isclose(angle, [0, 0, 0], atol=1e-5).all()
    assert np.isclose(t, [0, 0, 0]).all()


def test_distance_family_poses_rotoffset_sym():
    p2 = np.asarray(
        [[70, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [90, 90, 0, 0, 0, 0]], dtype=float
    )
    R_p2 = R.from_euler("XZX", p2[:, :3], degrees=True)
    R0 = R.random(random_state=0)
    R_p1 = R_p2 * R0
    p1 = np.zeros_like(p2)
    p1[:, :3] = R_p1.as_euler("XZX", degrees=True)
    angle, t = distance_family_poses(p1, p2)
    assert np.isclose(angle, [0, 0, 0], atol=1e-5).all()
    assert np.isclose(t, [0, 0, 0]).all()


def test_symmetrize_poses():
    poses, volume = (
        data.generated_isotropic()["poses"],
        data.generated_isotropic()["volumes"][0],
    )
    gt = data.generated_isotropic()["gt"]
    poses_sym = symmetrize_poses(poses, 9)
    mats = get_transform_matrix(
        volume.shape, poses_sym[0, :, :3], poses_sym[0, :, 3:], degrees=True
    )
    volume_rotated = affine_transform(np.stack((volume,) * 9), mats, batch=True)
    assert are_volumes_aligned(volume_rotated, gt).all()
