import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import affine_transform as affine_transform_cupy
from scipy.ndimage.interpolation import affine_transform
from scipy.spatial.transform import Rotation as R

from spfluo.utils import affine_transform as affine_transform_pytorch

interp_order = 3


def get_rot_vec_from_3d_rot_mat(rot_mat):
    r = R.from_matrix(rot_mat)
    rot_vec = r.as_euler("zxz", degrees=True)
    return rot_vec


def get_angle_from_2d_rot_mat(rot_mat):
    cos = rot_mat[0, 0]
    sin = rot_mat[1, 0]
    angle = np.arccos(cos)
    angle_degree = 180 * angle / np.pi
    if sin >= 0:
        return angle_degree
    else:
        return -angle_degree


def get_rot_vec_from_rot_mat(rot_mat):
    if rot_mat.shape == (2, 2):
        return get_angle_from_2d_rot_mat(rot_mat)
    elif rot_mat.shape == (3, 3):
        return get_rot_vec_from_3d_rot_mat(rot_mat)
    else:
        print("shape of rot mat must be either (2,2) or (3,3)")
        print("rot mat :", rot_mat)
        raise ValueError


def get_2d_rotation_matrix(angle):
    angle = np.pi * angle / 180
    rot_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return rot_mat


def point_cloud_rotation(point_cloud, rot_mat):
    return (rot_mat @ point_cloud.T).T


def get_rotation_matrix(rot_vec, convention="zxz"):
    if hasattr(rot_vec, "__iter__") and len(rot_vec) == 3:
        return get_3d_rotation_matrix(rot_vec, convention=convention)

    else:
        return get_2d_rotation_matrix(rot_vec)


def get_3d_rotation_matrix(rot_vec, convention="zxz"):
    """transforms the rotation vector into a rotation matrix"""

    r = R.from_euler(convention, rot_vec, degrees=True)
    rot_mat = r.as_matrix()
    return rot_mat


def translation(image, trans_vec, order=interp_order):
    nb_dim = len(image.shape)
    trans_vec = np.array(trans_vec)
    return affine_transform(
        image, np.eye(nb_dim), -trans_vec, order=order, mode="nearest"
    )


def rotation(volume, rot_mat, order=interp_order, trans_vec=None):
    """apply a rotation around center of image"""
    if trans_vec is None:
        trans_vec = np.zeros(len(volume.shape))
    c = np.array([size // 2 for size in volume.shape])
    rotated = affine_transform(
        volume, rot_mat.T, c - rot_mat.T @ (c + trans_vec), order=order, mode="constant"
    )
    return rotated, rot_mat


def rotation_gpu(volume, rot_mat, order=interp_order, trans_vec=None):
    """apply a rotation around center of image"""
    if trans_vec is None:
        trans_vec = cp.zeros(len(volume.shape))
    c = cp.array([size // 2 for size in volume.shape])
    rotated = affine_transform_cupy(
        volume, rot_mat.T, c - rot_mat.T @ (c + trans_vec), order=order, mode="constant"
    )
    return rotated, rot_mat


def rotation_gpu_pytorch(volume, rot_mat, order=1, trans_vec=None):
    """apply a rotation around center of image"""
    if order > 1:
        raise NotImplementedError("order should be 1")
    if trans_vec is None:
        trans_vec = volume.new_zeros((volume.size(0), volume.ndim - 2, 1))
    else:
        trans_vec = trans_vec[:, :, None]
    c = volume.new_tensor([size // 2 for size in volume.shape[2:]])[:, None]
    rotated = affine_transform_pytorch(
        volume,
        rot_mat.transpose(1, 2),
        (c - rot_mat.transpose(1, 2) @ (c + trans_vec))[..., 0],
        order=order,
        mode="zeros",
    )
    return rotated, rot_mat


def discretize_sphere_uniformly(nb_view_dir, nb_angles=20):
    """Generates a list of the two first euler angles that describe a uniform discretization of the sphere with the Fibonnaci sphere algorithm
    :param N: number of points
    """

    goldenRatio = (1 + 5**0.5) / 2
    i = np.arange(0, nb_view_dir)
    theta = np.mod(2 * np.pi * i / goldenRatio, 2 * np.pi)
    phi = np.arccos(1 - 2 * (i + 0.5) / nb_view_dir)
    psi = np.linspace(0, 2 * np.pi, nb_angles)
    """
    x, y, z = conversion_2_first_eulers_angles_cartesian(theta, phi, False)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)  # plot the point (2,3,4) on the figure
    plt.show()
    """
    return theta * 180 / np.pi, phi * 180 / np.pi, psi * 180 / np.pi


def conversion_2_first_eulers_angles_cartesian(theta, phi, degrees=True):
    if degrees:
        theta = theta * np.pi / 180
        phi = phi * np.pi / 180
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return x, y, z


def conversion_polar_cartesian(rho, theta, phi, degrees=False):
    if degrees:
        theta = theta * np.pi / 180
        phi = phi * np.pi / 180
    x, y, z = (
        rho * np.cos(theta) * np.sin(phi),
        rho * np.sin(theta) * np.sin(phi),
        rho * np.cos(phi),
    )
    return x, y, z


def conversion_cartesian_polar(x, y, z, degree=False):
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z / rho)
    if x > 0:
        theta = np.arctan(y / x)
    elif x < 0 and y >= 0:
        theta = np.arctan(y / x) + np.pi
    elif x < 0 and y < 0:
        theta = np.arctan(y / x) - np.pi
    elif x == 0 and y > 0:
        theta = np.pi / 2
    elif x == 0 and y < 0:
        theta = -np.pi / 2
    else:
        theta = None
    if degree:
        if theta is not None:
            theta = 180 * theta / np.pi
        phi = 180 * phi / np.pi
    return rho, theta, phi
