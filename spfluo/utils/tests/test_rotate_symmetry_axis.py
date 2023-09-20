import numpy as np
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from scipy.spatial.transform import Rotation as R

import spfluo.data as data
from spfluo.utils.rotate_symmetry_axis import (
    find_rot_mat_between_centriole_axis_and_z_axis,
)
from spfluo.utils.transform import get_transform_matrix_around_center
from spfluo.utils.volume import affine_transform


def distance_rotation_matrices(A, B):
    vec = np.array([1, 0, 0])
    return np.linalg.norm((A - B) @ vec)


def test_find_rot_mat_easy():
    centriole = data.generated_isotropic()["gt"]
    rot = find_rot_mat_between_centriole_axis_and_z_axis(centriole)
    assert np.isclose(distance_rotation_matrices(rot, np.identity(3)), 0, atol=1e-2)


@given(quaternions=arrays(float, (4,)))
def test_find_rot_mat(quaternions):
    assume(np.linalg.norm(quaternions) > 0)
    assume(not np.isinf(np.linalg.norm(quaternions)))
    quaternions /= np.linalg.norm(quaternions)
    rot = R.from_quat(quaternions).as_matrix()
    centriole = data.generated_isotropic()["gt"]
    centriole_rotated = affine_transform(
        centriole,
        np.linalg.inv(get_transform_matrix_around_center(centriole.shape, rot)),
    )
    rot_found = find_rot_mat_between_centriole_axis_and_z_axis(centriole_rotated)
    sym_x = np.eye(3)
    sym_x[0, 0] = -1
    rot_found_sym = sym_x @ rot_found.T
    assert np.isclose(
        distance_rotation_matrices(rot, rot_found.T), 0, atol=1e-2
    ) or np.isclose(distance_rotation_matrices(rot, rot_found_sym), 0, atol=1e-2)
