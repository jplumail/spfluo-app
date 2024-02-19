import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation as R

import spfluo.data as data
from spfluo.tests.helpers import assert_volumes_aligned
from spfluo.utils.rotate_symmetry_axis import (
    find_pose_from_z_axis_to_centriole_axis,
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


def create_data_random(quaternions, translation, translation_magnitude, create_data):
    _, (volumes, poses, gt) = create_data
    assume(np.linalg.norm(quaternions) > 0)
    assume(not np.isinf(np.linalg.norm(quaternions)))
    assume(np.linalg.norm(translation) > 0)
    assume(not np.isinf(np.linalg.norm(translation)))

    euler = R.from_quat(quaternions).as_euler("XZX", degrees=True)
    translation /= np.linalg.norm(translation)
    translation *= translation_magnitude * 5
    pose = np.concatenate((euler, translation))
    gt_rotated = affine_transform(
        gt, np.linalg.inv(get_transform_matrix_from_pose(volumes[0].shape, pose))
    )
    poses_rotated = compose_poses(invert_pose(pose), poses)
    return (volumes, poses_rotated, gt_rotated), (volumes, poses, gt), pose


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
    pose = find_pose_from_z_axis_to_centriole_axis(centriole, threshold=0.3)
    assert np.isclose(
        distance_rotation_matrices(
            R.from_euler("XZX", pose[:3], degrees=True).as_matrix(), np.identity(3)
        ),
        0,
        atol=1e-2,
    )


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    report_multiple_bugs=False,
    derandomize=True,
    print_blob=True,
)
@given(
    quaternions=st.tuples(
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
    ),
    translation=st.tuples(
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
    ),
    translation_magnitude=st.floats(0, 1, allow_infinity=False, allow_nan=False),
)
def test_find_pose(
    quaternions, translation, translation_magnitude, create_data, save_result
):
    (
        (volumes, poses, reconstruction),
        (_, poses_true_aligned, gt),
        true_pose,
    ) = create_data_random(quaternions, translation, translation_magnitude, create_data)
    pose = find_pose_from_z_axis_to_centriole_axis(reconstruction)

    poses_aligned = compose_poses(pose, poses)

    # assess errors
    err_rot1, err_trans1 = distance_poses(poses_aligned, poses_true_aligned, symmetry=9)
    err_rot1_corrected = err_rot1.copy()
    err_rot1_corrected[err_rot1 > 20] = err_rot1[err_rot1 > 20] - 40

    # centriole has another symmetry
    transformation_sym = np.asarray([0, 180, 0, 0, 0, 0], dtype=float)
    poses_aligned_sym = compose_poses(transformation_sym, poses_aligned)
    err_rot2, err_trans2 = distance_poses(
        poses_aligned_sym, poses_true_aligned, symmetry=9
    )
    err_rot2_corrected = err_rot2.copy()
    err_rot2_corrected[err_rot2 > 20] = err_rot2[err_rot2 > 20] - 40

    rot_error = 20
    trans_error = 1
    try:
        assert (
            np.isclose(err_rot1_corrected, 0, atol=rot_error).all()
            and np.isclose(err_trans1, 0, atol=trans_error).all()
        ) or (
            np.isclose(err_rot2_corrected, 0, atol=rot_error).all()
            and np.isclose(err_trans2, 0, atol=trans_error).all()
        )
    except AssertionError as e:
        reconstruction_aligned = affine_transform(
            reconstruction, get_transform_matrix_from_pose(volumes[0].shape, pose)
        )
        save_result(
            f"reconstruction_aligned-{'_'.join(map(str,np.round(true_pose,1).tolist()))}",
            reconstruction_aligned,
        )
        save_result("gt", gt)
        raise e


def test_apply_transformation_and_sym_to_poses(create_data, save_result):
    (
        (volumes, poses, reconstruction),
        (_, poses_true_aligned, reconstruction_true_aligned),
    ) = create_data

    pose_from_axis_to_reconstruction = find_pose_from_z_axis_to_centriole_axis(
        reconstruction
    )
    reconstruction_aligned = affine_transform(
        reconstruction,
        get_transform_matrix_from_pose(
            reconstruction.shape, pose_from_axis_to_reconstruction
        ),
    )
    save_result("reconstruction_aligned", reconstruction_aligned)
    assert_volumes_aligned(
        reconstruction_aligned, reconstruction_true_aligned, atol=0.5
    )

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
    assert_volumes_aligned(volumes_aligned, reconstruction_true_aligned, atol=0.5)

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
            volume_i_aligned_sym, reconstruction_true_aligned, atol=0.5
        )
