from typing import Tuple

from scipy.spatial.transform import Rotation

from ._array import Array, array_api_compat, cpu_only_compatibility


@cpu_only_compatibility
def euler_to_matrix(convention: str, euler_angles: Array, degrees=False) -> Array:
    return Rotation.from_euler(convention, euler_angles, degrees=degrees).as_matrix()


def get_transform_matrix(
    shape: Tuple[int, int, int],
    euler_angles: Array,
    translation: Array,
    convention: str = "XZX",
    degrees: bool = False,
):
    """
    Returns the transformation matrix in pixel coordinates.
    The transformation is the composition of a rotation (defined by 3 euler angles),
    and a translation (defined by a translation vector).
    The rotation is made around the center of the volume.
    Params:
        shape: Tuple[int, int, int]
            shape of the image to be transformed. D, H, W
        euler_angles: np.ndarray of shape ((N), 3)
            𝛗, 𝛉, 𝛙. See convention to see how they are used.
        translation: np.ndarray of shape ((N), 3)
        convention: str
            Euler angles convention in scipy terms.
            See `scipy.spatial.transform.Rotation`.
            Default to 'XZX'

                   a-------------b       numpy coordinates of points:
                  /             /|        - a = (0, 0, 0)
                 /             / |        - b = (0, 0, W-1)
                c-------------+  |        - c = (0, H-1, 0)
                |             |  |        - d = (D-1, H-1, 0)
                |             |  |        - e = (D-1, H-1, W-1)
                |             |  +
                |             X Y
                |             ↑↗
                d-----------Z←e   <-- reference frame used for rotations.
                                      The center of the rotation is at (D/2, H/2, W/2).

            If the convention 'XZX' is used:
                - first, rotate by 𝛗 around the X-axis. The XYZ frame is also rotated!
                - then, rotate by 𝛉 around the Z-axis.
                - finally, rotate by 𝛙 around the X-axis.

        degrees: bool
            Are the euler angles in degrees?
    Returns:
        np.ndarray of shape ((N), 4, 4)
        An (or N) affine tranformation(s) in homogeneous coordinates.
    """
    xp = array_api_compat.array_namespace(euler_angles, translation)
    array_kwargs = {"dtype": euler_angles.dtype, "device": xp.device(euler_angles)}
    rot = euler_to_matrix(convention, euler_angles, degrees=degrees)
    center = (xp.asarray(shape, **array_kwargs) - 1) / 2
    if len(euler_angles.shape) == 1:
        H_rot = xp.zeros((4, 4), **array_kwargs)
    elif len(euler_angles.shape) == 2:
        H_rot = xp.zeros((euler_angles.shape[0], 4, 4), **array_kwargs)
    H_rot[..., 3, 3] = 1.0
    H_center = xp.asarray(H_rot, copy=True)
    H_center[..., :3, 3] = -center  # 1. translation to (0,0,0)
    H_center[..., [0, 1, 2], [0, 1, 2]] = 1.0  # diag to 1
    H_rot[..., :3, :3] = rot  # 2. rotation
    H_rot[..., :3, 3] = (
        translation + center
    )  # 3. translation to center of image. 4. translation

    #   4-3-2 <- 1
    H = H_rot @ H_center
    return H


def distance_poses(
    p1: Array, p2: Array, convention: str = "XZX"
) -> Tuple[Array, Array]:
    """Compute the rotation distance and the euclidean distance between p1 and p2.
    Parameters:
        p1, p2 : arrays of shape (..., 6). Must be broadcastable.
            Represents poses (theta,psi,gamma,tz,ty,tx).
    Returns:
        distances : Tuple[Array, Array] of shape broadcasted dims.
    """
    # Rotation distance
    xp = array_api_compat.array_namespace(p1, p2)
    rot1, rot2 = xp.asarray(p1[..., :3]), xp.asarray(p2[..., :3])
    rot_mat1 = xp.reshape(
        euler_to_matrix(convention, xp.reshape(rot1, (-1, 3)), degrees=True),
        rot1.shape[:-1] + (3, 3),
    )
    rot_mat2 = xp.reshape(
        euler_to_matrix(convention, xp.reshape(rot2, (-1, 3)), degrees=True),
        rot2.shape[:-1] + (3, 3),
    )
    R = rot_mat1 @ xp.linalg.matrix_transpose(rot_mat2)
    rot_distance = xp.acos((R[..., [0, 1, 2], [0, 1, 2]].sum(-1) - 1) / 2) * 180 / xp.pi

    # Euclidian distance
    t1, t2 = p1[..., 3:], p2[..., 3:]
    trans_distance = xp.sum(((t1 - t2) ** 2), axis=-1) ** 0.5

    return rot_distance, trans_distance


def distance_family_poses(
    guessed_poses: Array, gt_poses: Array, convention: str = "XZX", symmetry: int = 1
):
    """Compute the rotation distance and the euclidean distance between guessed_poses
    and gt_poses.
    Account for an eventual offset in the guessed_poses.
        E.g. if the guessed poses are the same as the gt poses but rotated by R0, then
        the rotation distance will be 0.
    If symmetry is greater than 1, account also for symmetries.
        The symmetry must be around the first axis.

    Parameters:
        guessed_poses, gt_poses : arrays of shape (N, 6).
            Represents poses (theta,psi,gamma,tz,ty,tx).
        convention : string
        symmetry : int
    Returns:
        rotation distance, translation distance : Tuple[Array, Array] of shape (N,)
    """
    xp = array_api_compat.array_namespace(guessed_poses, gt_poses)

    # Rotation distances
    # 1. convert euler angles to matrices
    euler1, euler2 = xp.asarray(guessed_poses[:, :3]), xp.asarray(gt_poses[:, :3])
    N, _ = euler1.shape
    guessed_rot_mat = euler_to_matrix(convention, euler1, degrees=True)
    gt_rot_mat = euler_to_matrix(convention, euler2, degrees=True)
    sym_euler = xp.zeros(
        (symmetry, 3), dtype=guessed_rot_mat.dtype, device=xp.device(guessed_rot_mat)
    )
    sym_euler[:, 0] = (
        -2
        * xp.arange(
            symmetry, dtype=guessed_rot_mat.dtype, device=xp.device(guessed_rot_mat)
        )
        * xp.pi
        / symmetry
    )
    sym_matrices = euler_to_matrix(convention, sym_euler, degrees=False)

    basis_change = (
        xp.linalg.matrix_transpose(guessed_rot_mat[None, :] @ sym_matrices[:, None])
        @ gt_rot_mat[None, :]
    )  # shape (s, N, 3, 3)
    assert basis_change.shape == (symmetry, N, 3, 3)
    diff = (
        guessed_rot_mat[None, :, None]  # shape (1, N, 1, 3, 3)
        @ basis_change[:, None]  # shape (s, 1, N, 3, 3)
        @ xp.linalg.matrix_transpose(gt_rot_mat[None, :, None])  # shape (1, N, 1, 3, 3)
    )
    # shape (s, N, N, 3, 3)
    traces = xp.sum(diff[:, :, :, [0, 1, 2], [0, 1, 2]], axis=-1)  # shape (s, N, N)
    traces[traces > 3.0] = 3.0
    angles = xp.acos((traces - 1) / 2)
    angles = xp.min(angles, axis=0)  # shape (N, N, 3, 3)
    mean_angles = xp.mean(
        angles,
        axis=-1,
    )
    mean_angles = mean_angles * 180 / xp.pi

    # Translation distances, simple L2 norm
    t1, t2 = xp.asarray(guessed_poses[..., 3:]), xp.asarray(gt_poses[..., 3:])
    trans_distances = xp.sum(((t1 - t2) ** 2), axis=-1) ** 0.5

    return mean_angles, trans_distances


def get_zoom_matrix(
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    xp,
    **array_kwargs,
):
    in_shape, out_shape = xp.asarray(input_shape, **array_kwargs), xp.asarray(
        output_shape, **array_kwargs
    )
    input_center, output_center = (in_shape - 1) / 2, (out_shape - 1) / 2
    H_center, H_homo = xp.eye(4, **array_kwargs), xp.eye(4, **array_kwargs)
    H_center[:3, 3] = -input_center  # 1. translation to (0,0,0)
    H_homo[:3, 3] = output_center  # 3. translation to center of image

    #    3-2 <- 1
    H = H_homo @ H_center
    return H
