from typing import Optional

import numpy as np
import tifffile
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

from spfluo.utils.array import Array, array_namespace
from spfluo.utils.loading import read_poses, save_poses
from spfluo.utils.transform import compose_poses, get_transform_matrix_from_pose
from spfluo.utils.volume import affine_transform

DEFAULT_THRESHOLD = 0.3


def convert_im_to_point_cloud(im: Array, thesh: float):
    xp = array_namespace(im)
    coordinates = xp.where(im >= thesh)
    coordinates = xp.stack(coordinates, axis=-1)
    return coordinates


def skew_symmetric_cross_product(v: Array):
    v1, v2, v3 = v[0], v[1], v[2]
    xp = array_namespace(v)
    return xp.asarray([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])


def find_rotation_between_two_vectors(a: Array, b: Array):
    """returns the rotation matrix that rotates vector a onto vector b
    (the rotation matrix s.t. Ra = b)"""
    xp = array_namespace(a, b)
    v = xp.linalg.cross(a, b)
    s = xp.linalg.vector_norm(v)
    c = xp.linalg.vecdot(a, b)
    ssc = skew_symmetric_cross_product(v)
    rot = xp.eye(3) + ssc + (ssc @ ssc) * (1 - c) / s**2
    return rot


def find_centriole_symmetry_axis(
    centriole_im: Array, threshold: float = DEFAULT_THRESHOLD
):
    xp = array_namespace(centriole_im)
    ma = np.max(centriole_im)
    centriol_pc = convert_im_to_point_cloud(centriole_im, ma * threshold)
    pca = PCA(n_components=3)
    pca.fit(centriol_pc)
    sum_of_2_by_2_differences = xp.zeros(3)
    for i in range(3):
        for j in range(3):
            if j != i:
                sum_of_2_by_2_differences[i] += xp.abs(
                    pca.singular_values_[i] - pca.singular_values_[j]
                )
    idx_dim_pca = xp.argmax(sum_of_2_by_2_differences)
    symmetry_axis = xp.asarray(pca.components_[idx_dim_pca])
    return xp.asarray(symmetry_axis), xp.asarray(pca.mean_)


def find_pose_from_z_axis_to_centriole_axis(
    centriole_im: Array, axis_indice=0, threshold=DEFAULT_THRESHOLD, convention="XZX"
):
    """Find the pose of the transformation from the axis to the centriole"""
    xp = array_namespace(centriole_im)
    symmetry_axis, center = find_centriole_symmetry_axis(
        centriole_im, threshold=threshold
    )
    z_axis = xp.asarray([0, 0, 0])
    z_axis[axis_indice] = 1
    rot = find_rotation_between_two_vectors(symmetry_axis, z_axis)
    trans = (xp.asarray(centriole_im.shape) - 1) / 2 - center
    pose = xp.zeros((6,), dtype=xp.float64)
    pose[:3] = R.from_matrix(rot.T).as_euler(convention, degrees=True)
    pose[3:] = -trans
    return pose


def main(
    volume_path: str,
    convention: str = "XZX",
    threshold: float = DEFAULT_THRESHOLD,
    output_volume_path: Optional[str] = None,
    poses_path: Optional[str] = None,
    output_poses_path: Optional[str] = None,
):
    volume = tifffile.imread(volume_path)
    pose = find_pose_from_z_axis_to_centriole_axis(volume, threshold=threshold)
    if output_volume_path:
        rotated_volume = affine_transform(
            volume,
            get_transform_matrix_from_pose(pose),
        )
        tifffile.imwrite(output_volume_path, rotated_volume)
    if poses_path:
        poses, names = read_poses(poses_path)
        new_poses = compose_poses(pose, poses, convention=convention)
        save_poses(output_poses_path, new_poses, names)
