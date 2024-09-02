import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import spfluo.data as data
from spfluo.tests.helpers import assert_volumes_aligned
from spfluo.utils.rotate_symmetry_axis import (
    find_pose_from_z_axis_centered_to_centriole_axis,
)
from spfluo.utils.transform import (
    compose_poses,
    distance_poses,
    get_transform_matrix_from_pose,
    invert_pose,
    symmetrize_poses,
)
from spfluo.utils.volume import (
    affine_transform,
)

EPS = 5  # angle error margin in degrees


def distance_transformations(rot1, trans1, rot2, trans2):
    pose1 = np.concatenate((R.from_matrix(rot1).as_euler("XZX", degrees=True), trans1))
    pose2 = np.concatenate((R.from_matrix(rot2).as_euler("XZX", degrees=True), trans2))
    return distance_poses(pose1, pose2, symmetry=9, ignore_symmetry=False)


def distance_rotation_matrices(A, B):
    vec = np.array([1, 0, 0])
    return np.linalg.norm((A - B) @ vec)


@pytest.fixture(scope="module")
def create_data():
    d = data.generated_anisotropic()
    volumes, poses, gt, _ = d["volumes"], d["poses"], d["gt"], d["psf"]
    pose = np.asarray([45, 45, 0, 5, -5, 2], dtype=float)
    gt_rotated = affine_transform(
        gt, np.linalg.inv(get_transform_matrix_from_pose(volumes[0].shape, pose))
    )
    poses_rotated = compose_poses(invert_pose(pose), poses)
    return (volumes, poses_rotated, gt_rotated), (volumes, poses, gt)


def test_data(create_data, save_result):
    (volumes, poses, reconstruction), _ = create_data
    # Vérification que les données sont bonnes
    volumes_rotated_back = affine_transform(
        volumes, get_transform_matrix_from_pose(volumes[0].shape, poses), batch=True
    )
    save_result("volumes_rotated_back", volumes_rotated_back)
    save_result("gt_rotated", reconstruction)
    assert_volumes_aligned(reconstruction, volumes_rotated_back, atol=1)


def test_find_rot_mat_easy(create_data):
    _, (_, _, centriole) = create_data
    pose = find_pose_from_z_axis_centered_to_centriole_axis(centriole, 9)
    assert np.isclose(
        distance_rotation_matrices(
            R.from_euler("XZX", pose[:3], degrees=True).as_matrix(), np.identity(3)
        ),
        0,
        atol=1e-1,
    )


def test_apply_transformation_and_sym_to_poses(create_data, save_result):
    (
        (volumes, poses, reconstruction),
        (_, poses_true_aligned, reconstruction_true_aligned),
    ) = create_data

    pose_from_axis_to_reconstruction = find_pose_from_z_axis_centered_to_centriole_axis(
        reconstruction, 9, center_precision=0.5
    )
    reconstruction_aligned = affine_transform(
        reconstruction,
        get_transform_matrix_from_pose(
            reconstruction.shape, pose_from_axis_to_reconstruction
        ),
    )
    save_result("reconstruction", reconstruction)
    save_result("reconstruction_true_aligned", reconstruction_true_aligned)
    save_result("reconstruction_aligned", reconstruction_aligned)
    assert_volumes_aligned(reconstruction_aligned, reconstruction_true_aligned, atol=1)

    # Test poses_aligned are ok
    poses_from_reconstruction_to_vols = poses
    poses_aligned = compose_poses(
        pose_from_axis_to_reconstruction, poses_from_reconstruction_to_vols
    )
    volumes_aligned = affine_transform(
        volumes,
        get_transform_matrix_from_pose(volumes[0].shape, poses_aligned),
        batch=True,
    )
    save_result("volumes_aligned", volumes_aligned)
    assert_volumes_aligned(volumes_aligned, reconstruction_true_aligned, atol=1)
    angular_errors, trans_errors = distance_poses(
        poses_aligned, poses_true_aligned, symmetry=9
    )
    bottom_up_pose = np.asarray(
        [0, 180, 0, 0, 0, 0]
    )  # alignement can put the image upside down
    poses_aligned_flipped = compose_poses(bottom_up_pose, poses_aligned)
    volumes_aligned = affine_transform(
        volumes,
        get_transform_matrix_from_pose(volumes[0].shape, poses_aligned_flipped),
        batch=True,
    )
    save_result("volumes_aligned_flipped", volumes_aligned)
    angular_errors_flipped, _ = distance_poses(
        poses_aligned_flipped, poses_true_aligned, symmetry=9
    )
    assert (
        all(np.isclose(angular_errors, 0, atol=360 / 9 / 2 + EPS))
        or all(np.isclose(angular_errors_flipped, 0, atol=360 / 9 / 2 + EPS))
    ) and all(np.isclose(trans_errors, 0, atol=1.5))

    # Test symmetrizing these poses
    poses_aligned_sym = symmetrize_poses(poses_aligned, 9)
    volume0_aligned_sym = affine_transform(
        np.stack((volumes[0],) * 9),
        get_transform_matrix_from_pose(volumes[0].shape, poses_aligned_sym[0]),
        batch=True,
    )
    save_result("volume0_aligned_sym", volume0_aligned_sym, metadata={"axes": "TZYX"})

    for i in range(volumes.shape[0]):
        volume_i_aligned_sym = affine_transform(
            np.stack((volumes[i],) * 9),
            get_transform_matrix_from_pose(volumes[0].shape, poses_aligned_sym[i]),
            batch=True,
        )
        assert_volumes_aligned(
            volume_i_aligned_sym, reconstruction_true_aligned, atol=1
        )


@pytest.mark.skip(reason="to fix")
def test_real_data(save_result):
    d = data.real_ab_initio_reconstruction()
    image, pose = d["reconstruction"], d["pose"]

    aligned_image = affine_transform(
        image, get_transform_matrix_from_pose(image.shape[:3], pose)
    )

    pose_estimate = find_pose_from_z_axis_centered_to_centriole_axis(
        image, 9, center_precision=1
    )

    bottom_up_pose = np.asarray(
        [0, 180, 0, 0, 0, 0]
    )  # alignement can put the image upside down
    pose_estimate = compose_poses(bottom_up_pose, pose_estimate)
    aligned_image2 = affine_transform(
        image, get_transform_matrix_from_pose(image.shape[:3], pose_estimate)
    )

    save_result("image", image)
    save_result("aligned_image", aligned_image)
    save_result("aligned_image_estimate", aligned_image2)

    angular_distance, trans_distance = distance_poses(pose, pose_estimate, symmetry=9)

    print(pose, pose_estimate)
    np.testing.assert_allclose(angular_distance, 0, atol=360 / 9 / 2 + EPS)
    np.testing.assert_allclose(trans_distance, 0, atol=2)
