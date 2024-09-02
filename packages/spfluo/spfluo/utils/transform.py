from functools import partial
from typing import TYPE_CHECKING, Tuple

from scipy.spatial.transform import Rotation

from .array import array_namespace, get_device, median, numpy_only_compatibility

if TYPE_CHECKING:
    from spfluo.utils.array import Array


@numpy_only_compatibility
def euler_to_matrix(convention: str, euler_angles: "Array", degrees=False) -> "Array":
    """
    Params:
        convention: str
        euler_angles: Array of shape (3,) or (N, 3)
        degrees: bool
    """
    return Rotation.from_euler(convention, euler_angles, degrees=degrees).as_matrix()


@numpy_only_compatibility
def matrix_to_euler(convention: str, matrix: "Array", degrees: bool = False):
    """
    Params:
        convention: str
        euler_angles: Array of shape (3,) or (N, 3)
        degrees: bool
    """
    return Rotation.from_matrix(matrix).as_euler(convention, degrees=degrees)


def compose_poses(pose1: "Array", pose2: "Array", convention="XZX"):
    """Returns a pose: pose = pose2 ○ pose1
    Params:
        pose1, pose2: Array of shape (3,) or (N, 3)
        convention: str
    """
    xp = array_namespace(pose1, pose2)
    to_matrix = partial(euler_to_matrix, convention, degrees=True)
    to_euler = partial(matrix_to_euler, convention, degrees=True)
    new_pose = xp.zeros_like(pose1) + xp.zeros_like(pose2)
    new_pose[..., :3] = to_euler(to_matrix(pose2[..., :3]) @ to_matrix(pose1[..., :3]))
    new_pose[..., 3:] = (
        pose2[..., 3:] + (to_matrix(pose2[..., :3]) @ pose1[..., 3:, None])[..., 0]
    )
    return new_pose


def invert_pose(pose: "Array", convention="XZX"):
    xp = array_namespace(pose)
    inv_mat = xp.linalg.matrix_transpose(
        euler_to_matrix(convention, pose[..., :3], degrees=True)
    )
    t = pose[..., 3:]
    new_pose = xp.zeros_like(pose)
    new_pose[..., :3] = matrix_to_euler(convention, inv_mat, degrees=True)
    new_pose[..., 3:] = -(xp.astype(inv_mat, t.dtype) @ t[..., None])[..., 0]
    return new_pose


def get_transform_matrix_around_center(
    shape: Tuple[int, ...], rotation_matrix: "Array"
):
    """
    Params:
        - shape
        - rotation_matrix: Array of shape (..., ndim, ndim)
    Returns:
        Array of shape (..., ndim+1, ndim+1)
    """
    xp = array_namespace(rotation_matrix)
    array_kwargs = {
        "dtype": rotation_matrix.dtype,
        "device": get_device(rotation_matrix),
    }
    assert rotation_matrix.ndim >= 2
    ndim = rotation_matrix.shape[-1]
    assert ndim == rotation_matrix.shape[-2]
    assert ndim == len(shape)
    center = (xp.asarray(shape, **array_kwargs) - 1) / 2
    ndim = rotation_matrix.shape[-1]
    H_rot = xp.zeros(
        (rotation_matrix.shape[:-2]) + (ndim + 1, ndim + 1), **array_kwargs
    )
    H_rot[..., ndim, ndim] = 1.0
    H_center = xp.asarray(H_rot, copy=True)
    H_center[..., :ndim, ndim] = -center  # 1. translation to (0,0,0)
    for i in range(ndim):
        H_center[..., i, i] = 1.0  # diag to 1
    H_rot[..., :ndim, :ndim] = rotation_matrix  # 2. rotation
    H_rot[..., :ndim, ndim] = center  # 3. translation to center of image.

    #   3-2 <- 1
    H = H_rot @ H_center
    return H


def get_transform_matrix(
    shape: Tuple[int, int, int],
    euler_angles: "Array",
    translation: "Array",
    convention: str = "XZX",
    degrees: bool = False,
    dtype: str = "float64",
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
    xp = array_namespace(euler_angles, translation)
    dtype = getattr(xp, dtype)
    assert xp.isdtype(dtype, "real floating")
    rot = euler_to_matrix(convention, euler_angles, degrees=degrees)
    H_rot = get_transform_matrix_around_center(shape, rot)
    H_rot[..., :3, 3] += translation  # adds the translation

    return xp.asarray(H_rot, dtype=dtype)


def get_transform_matrix_from_pose(
    shape: Tuple[int, int, int],
    pose: "Array",
    *,
    convention: str = "XZX",
):
    return get_transform_matrix(
        shape, pose[..., :3], pose[..., 3:], convention=convention, degrees=True
    )


def symmetrize_angles(
    euler_angles: "Array", symmetry: int, convention: str = "XZX", degrees: bool = False
) -> "Array":
    """
    axis of symmetry is around the X-axis (see get_transform_matrix)
    the euler_angles are in the XZX convention, so that the third angle is purely the
    angle of symmetry.
    Params:
        - euler_angles: Array of shape (..., 3)
        - symmetry: int
            degree of the symmetry
    Returns:
        euler_angles_sym: Array of shape (..., symmetry, 3)
    """
    assert convention == "XZX"
    xp = array_namespace(euler_angles)
    full_range = 360 if degrees else 2 * xp.pi
    symmetry_offset = xp.asarray(
        [i * full_range / symmetry for i in range(symmetry)],
        device=get_device(euler_angles),
        dtype=euler_angles.dtype,
    )  # shape (symmetry,)
    euler_angles_sym = xp.asarray(euler_angles, copy=True)
    euler_angles_sym = xp.stack(
        (euler_angles,) * symmetry, axis=-2
    )  # shape (..., symmetry, 3)
    euler_angles_sym[..., 2] += symmetry_offset
    return euler_angles_sym


def symmetrize_poses(poses: "Array", symmetry: int, convention: str = "XZX") -> "Array":
    """
    Params:
        - poses: Array of shape (..., 6)
        - symmetry: int
            degree of the symmetry
    Returns:
        - poses_sym: shape (..., k, 6)
    """
    assert convention == "XZX"
    euler_angles_sym = symmetrize_angles(
        poses[..., :3], symmetry=symmetry, degrees=True
    )  # shape (..., k, 3)
    xp = array_namespace(euler_angles_sym)
    poses_sym = xp.concat(
        (euler_angles_sym, xp.zeros_like(euler_angles_sym)), axis=-1
    )  # shape (..., k, 6)
    poses_sym[..., 3:] = poses[..., None, 3:]
    return poses_sym


def distance_poses(
    p1: "Array", p2: "Array", convention: str = "XZX", symmetry: int = 1
):
    """Compute the rotation distance and the euclidean distance between p1 and p2.
    Parameters:
        p1, p2 : arrays of shape (..., 6). Must be broadcastable.
            Represents poses (theta,psi,gamma,tz,ty,tx).
    Returns:
        distances : Tuple[Array, Array] of shape broadcasted dims.
    """
    # Rotation distance
    xp = array_namespace(p1, p2)
    dtype, device = p1.dtype, get_device(p1)
    euler1, euler2 = xp.asarray(p1[..., :3]), xp.asarray(p2[..., :3])
    rot_mat1 = xp.reshape(
        euler_to_matrix(convention, xp.reshape(euler1, (-1, 3)), degrees=True),
        euler1.shape[:-1] + (3, 3),
    )  # shape (..., 3, 3)
    rot_mat2 = xp.reshape(
        euler_to_matrix(convention, xp.reshape(euler2, (-1, 3)), degrees=True),
        euler2.shape[:-1] + (3, 3),
    )  # shape (..., 3, 3)
    sym_euler = xp.zeros((symmetry, 3), dtype=dtype, device=device)
    sym_euler[:, 2] = (
        -2 * xp.arange(symmetry, dtype=dtype, device=device) * xp.pi / symmetry
    )
    sym_matrices = euler_to_matrix(
        convention, sym_euler, degrees=False
    )  # shape (s, 3, 3)
    R = (rot_mat1[..., None, :, :] @ sym_matrices) @ xp.linalg.matrix_transpose(
        rot_mat2[..., None, :, :]
    )  # shape (..., s, 3, 3)
    traces = xp.sum(R[..., [0, 1, 2], [0, 1, 2]], axis=-1)  # shape (..., s)
    traces[traces > 3.0] = 3.0
    traces[traces < -1.0] = -1.0
    angles = xp.acos((traces - 1) / 2) * 180 / xp.pi  # shape (..., s)
    best_s = xp.argmin(xp.abs(xp.mean(angles, axis=tuple(range(angles.ndim - 1)))))
    rot_distance = angles[..., best_s]  # shape (...)

    # Euclidian distance
    t1, t2 = p1[..., 3:], p2[..., 3:]
    trans_distance = xp.sum(((t1 - t2) ** 2), axis=-1) ** 0.5

    return rot_distance, trans_distance


def distance_family_poses(
    guessed_poses: "Array",
    gt_poses: "Array",
    convention: str = "XZX",
    symmetry: int = 1,
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
    xp = array_namespace(guessed_poses, gt_poses)

    # Rotation distances
    # 1. convert euler angles to matrices
    euler1, euler2 = xp.asarray(guessed_poses[:, :3]), xp.asarray(gt_poses[:, :3])
    N, _ = euler1.shape
    guessed_rot_mat = euler_to_matrix(convention, euler1, degrees=True)
    gt_rot_mat = euler_to_matrix(convention, euler2, degrees=True)
    sym_euler = xp.zeros(
        (symmetry, 3), dtype=guessed_rot_mat.dtype, device=get_device(guessed_rot_mat)
    )
    sym_euler[:, 0] = (
        -2
        * xp.arange(
            symmetry, dtype=guessed_rot_mat.dtype, device=get_device(guessed_rot_mat)
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
    angles = xp.min(angles, axis=0)  # shape (s, N, N)
    median_angles = median(angles, axis=-1, xp=xp)
    median_angles = median_angles * 180 / xp.pi

    # Translation distances, simple L2 norm
    t1, t2 = xp.asarray(guessed_poses[..., 3:]), xp.asarray(gt_poses[..., 3:])
    trans_distances = xp.sum(((t1 - t2) ** 2), axis=-1) ** 0.5

    return median_angles, trans_distances
