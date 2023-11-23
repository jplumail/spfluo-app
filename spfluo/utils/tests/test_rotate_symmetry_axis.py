import numpy as np
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from scipy.spatial.transform import Rotation as R

import spfluo.data as data
from spfluo.utils.rotate_symmetry_axis import (
    apply_rot_to_poses,
    find_rot_mat_between_centriole_axis_and_z_axis,
)
from spfluo.utils.transform import (
    get_transform_matrix,
    get_transform_matrix_around_center,
    symmetrize_poses,
)
from spfluo.utils.volume import affine_transform, are_volumes_aligned


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


def test_apply_rot_to_poses():
    d = data.generated_anisotropic()
    volumes, poses, gt, _ = d["volumes"], d["poses"], d["gt"], d["psf"]
    rot = R.from_euler("XZX", [45, 45, 0], degrees=True).as_matrix()
    gt_rotated = affine_transform(
        gt, np.linalg.inv(get_transform_matrix_around_center(volumes[0].shape, rot))
    )
    poses_rotated = apply_rot_to_poses(rot, poses)

    # Vérification que les données sont bonnes
    # volumes_rotated_back = affine_transform(
    #   volumes,
    #   get_transform_matrix(
    #       volumes[0].shape,
    #       poses_rotated[:,:3],
    #       poses_rotated[:,3:],
    #       degrees=True
    #   ),
    #   batch=True
    # )
    # import napari
    # v = napari.view_image(volumes_rotated_back) # alignés avec gt_rotated
    # v.add_image(gt_rotated) # gt mais rotated
    # napari.run()

    gt = gt_rotated
    poses = poses_rotated

    rot2 = find_rot_mat_between_centriole_axis_and_z_axis(gt)
    poses_aligned = apply_rot_to_poses(rot2, poses)
    gt_aligned = affine_transform(
        gt,
        np.linalg.inv(get_transform_matrix_around_center(gt.shape, rot2)),
    )
    assert are_volumes_aligned(
        gt_aligned, d["gt"], atol=0.1
    )  # d["gt"] aligné avec l'axe

    # Vérification que les poses_aligned sont bonnes
    # import napari
    # v = napari.view_image(gt_aligned)
    # volumes_aligned = affine_transform(
    #     volumes,
    #     get_transform_matrix(
    #         volumes[0].shape,
    #         poses_aligned[:,:3],
    #         poses_aligned[:,3:],
    #         degrees=True
    #     ),
    #     batch=True
    # )
    # v.add_image(volumes_aligned)
    # napari.run()

    poses_aligned_sym = symmetrize_poses(poses_aligned, 9)

    # Vérification que les poses_aligned_sym sont bien aligned et sym!
    # volume0_aligned_sym = affine_transform(
    #     np.stack((volumes[0],)*9),
    #     get_transform_matrix(
    #         volumes[0].shape,
    #         poses_aligned_sym[0,:,:3],
    #         poses_aligned_sym[0,:,3:],
    #         degrees=True
    #     ),
    #     batch=True
    # )
    # import napari
    # v = napari.view_image(volume0_aligned_sym)
    # v.add_image(gt_aligned)
    # v.add_image(gt)
    # napari.run()

    for i in range(volumes.shape[0]):
        volume_i_aligned_sym = affine_transform(
            np.stack((volumes[i],) * 9),
            get_transform_matrix(
                volumes[0].shape,
                poses_aligned_sym[i, :, :3],
                poses_aligned_sym[i, :, 3:],
                degrees=True,
            ),
            batch=True,
        )
        assert are_volumes_aligned(volume_i_aligned_sym, d["gt"], atol=0.1).all()
