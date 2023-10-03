from typing import Optional

import numpy as np
import tifffile
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

from spfluo.utils.loading import read_poses, save_poses
from spfluo.utils.transform import get_transform_matrix_around_center
from spfluo.utils.volume import affine_transform


def convert_im_to_point_cloud(im, thesh):
    coordinates = np.where(im >= thesh)
    coordinates = np.array(coordinates).T
    return coordinates


def skew_symmetric_cross_product(v):
    v1, v2, v3 = v[0], v[1], v[2]
    return np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])


def find_rotation_between_two_vectors(a, b):
    """returns the rotation matrix that rotates vector a onto vector b
    (the rotation matrix s.t. Ra = b)"""
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    ssc = skew_symmetric_cross_product(v)
    rot = np.eye(3) + ssc + ssc.dot(ssc) * (1 - c) / s**2
    return rot


def find_centriole_symmetry_axis(centriole_im):
    ma = np.max(centriole_im) / 2
    centriol_pc = convert_im_to_point_cloud(centriole_im, ma / 3)
    pca = PCA(n_components=3)
    pca.fit(centriol_pc)
    sum_of_2_by_2_differences = np.zeros(3)
    for i in range(3):
        for j in range(3):
            if j != i:
                sum_of_2_by_2_differences[i] += np.abs(
                    pca.singular_values_[i] - pca.singular_values_[j]
                )
    idx_dim_pca = np.argmax(sum_of_2_by_2_differences)
    symmetry_axis = pca.components_[idx_dim_pca]
    return symmetry_axis


def find_rot_mat_between_centriole_axis_and_z_axis(centriole_im, axis_indice=0):
    symmetry_axis = find_centriole_symmetry_axis(centriole_im)
    z_axis = np.array([0, 0, 0])
    z_axis[axis_indice] = 1
    rot = find_rotation_between_two_vectors(symmetry_axis, z_axis)
    return rot


def rotate_centriole_to_have_symmetry_axis_along_z_axis(centriole_im, axis_indice=0):
    rot = find_rot_mat_between_centriole_axis_and_z_axis(centriole_im, axis_indice)
    rotated_im = affine_transform(
        centriole_im,
        np.linalg.inv(get_transform_matrix_around_center(centriole_im.shape, rot)),
    )
    return rotated_im


def apply_rot_to_poses(rot, poses, convention="XZX"):
    rotation_to_axis_from_volume = R.from_matrix(rot)
    rotations_to_particles_from_volume = R.from_euler(
        convention, poses[:, :3], degrees=True
    )
    rotations_to_particles_from_axis = (
        rotations_to_particles_from_volume * rotation_to_axis_from_volume.inv()
    )
    new_poses = poses.copy()
    new_poses[:, :3] = rotations_to_particles_from_axis.as_euler(
        convention, degrees=True
    )
    return new_poses


def main(
    volume_path: str,
    convention: str = "XZX",
    output_volume_path: Optional[str] = None,
    poses_path: Optional[str] = None,
    output_poses_path: Optional[str] = None,
):
    volume = tifffile.imread(volume_path)
    rot = find_rot_mat_between_centriole_axis_and_z_axis(volume)
    print(tuple(R.from_matrix(rot).as_euler(convention, degrees=True)))
    if output_volume_path:
        rotated_volume = affine_transform(
            volume,
            np.linalg.inv(get_transform_matrix_around_center(volume.shape, rot)),
        )
        tifffile.imwrite(output_volume_path, rotated_volume)
    if poses_path:
        assert output_poses_path
        poses, names = read_poses(poses_path)
        new_poses = apply_rot_to_poses(rot, poses, convention=convention)
        save_poses(output_poses_path, new_poses, names)
