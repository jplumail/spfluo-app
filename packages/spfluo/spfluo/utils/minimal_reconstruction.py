import typing

from spfluo.refinement.refinement import create_poses_grid, reconstruction_L2
from spfluo.utils.array import array_namespace
from spfluo.utils.transform import (
    compose_poses,
    get_transform_matrix_from_pose,
    symmetrize_poses,
)
from spfluo.utils.volume import affine_transform, phase_cross_correlation

if typing.TYPE_CHECKING:
    from spfluo.utils.array import Array


def reconstruct(
    centriole_top: "Array",
    centriole_side: "Array",
    pose_side: "Array",
    psf: "Array",
    /,
    lambda_=1.0,
    M=100,
):
    xp = array_namespace(centriole_top, centriole_side)
    pose_top = xp.asarray(
        [0, 0, 0, 0, 0, 0], dtype=xp.float64
    )  # top particle is the reference

    assert centriole_side.shape == centriole_side.shape == psf.shape
    assert centriole_side.ndim == 3 or centriole_side.ndim == 4

    multichannel = centriole_side.ndim == 4

    # Find translation between centriole side and top
    centriole_side_rotated_back = affine_transform(
        centriole_side,
        get_transform_matrix_from_pose(centriole_side.shape, pose_side),
        multichannel=multichannel,
    )
    shift, error, phasediff = phase_cross_correlation(
        centriole_top,
        centriole_side_rotated_back,
        upsample_factor=10,
        normalization=None,
        multichannel=multichannel,
    )
    pose_reference_to_centriole_side_rotated_back = xp.concat(
        (xp.asarray([0, 0, 0], dtype=xp.float64), -shift)
    )
    pose_side_registered = compose_poses(
        pose_reference_to_centriole_side_rotated_back, pose_side
    )

    poses_symmetry, _ = create_poses_grid(xp, 1, M, symmetry=9)
    poses_side_registered_symmetry = compose_poses(poses_symmetry, pose_side_registered)

    centriole_side_registered_sym = affine_transform(
        xp.stack((centriole_side,) * M),
        get_transform_matrix_from_pose(
            centriole_side.shape, poses_side_registered_symmetry
        ),
        batch=True,
        multichannel=multichannel,
    )

    _, errors, _ = phase_cross_correlation(
        centriole_top,
        centriole_side_registered_sym,
        upsample_factor=1,
        normalization=None,
        nb_spatial_dims=3,
        multichannel=multichannel,
    )
    best = xp.argmin(errors)
    best_pose_side = poses_side_registered_symmetry[best]

    # reconstruct
    poses = xp.stack(
        (symmetrize_poses(best_pose_side, 9), symmetrize_poses(pose_top, 9))
    )
    poses = xp.permute_dims(poses, (1, 0, 2))
    return reconstruction_L2(
        xp.stack((centriole_side, centriole_top)),
        psf,
        poses,
        lambda_=xp.asarray(lambda_),
        device="cpu",
        symmetry=True,
    )
