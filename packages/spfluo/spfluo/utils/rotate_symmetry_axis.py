from __future__ import annotations

import logging
import math
from functools import partial
from typing import TYPE_CHECKING, Optional

import numpy as np
import tifffile
from ome_types import from_xml
from ome_types.model.simple_types import UnitsLength
from sklearn.decomposition import PCA

from spfluo.refinement.refinement import create_poses_grid
from spfluo.utils.array import (
    array_namespace,
    get_device,
    is_cupy_array,
    is_numpy_array,
    to_numpy,
)
from spfluo.utils.loading import get_data_from_ome_tiff, read_poses, save_poses
from spfluo.utils.transform import (
    compose_poses,
    get_transform_matrix_around_center,
    get_transform_matrix_from_pose,
    invert_pose,
    matrix_to_euler,
)
from spfluo.utils.volume import affine_transform, interpolate_to_size, resample

logger = logging.getLogger("spfluo.utils.rotate_symmetry_axis")

if TYPE_CHECKING:
    from spfluo.utils.array import Array
DEFAULT_THRESHOLD = 0.3


def convert_im_to_point_cloud(im: "Array", thesh: float):
    xp = array_namespace(im)
    coordinates = xp.nonzero(im >= thesh)
    coordinates = xp.stack(coordinates, axis=-1)
    return coordinates


def skew_symmetric_cross_product(v: "Array"):
    v1, v2, v3 = v[0], v[1], v[2]
    xp = array_namespace(v)
    return xp.asarray(
        [[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]], dtype=v.dtype, device=get_device(v)
    )


def find_rotation_between_two_vectors(a: "Array", b: "Array"):
    """returns the rotation matrix that rotates vector a onto vector b
    (the rotation matrix s.t. Ra = b)"""
    xp = array_namespace(a, b)
    device = get_device(a)
    dtype = a.dtype
    v = xp.linalg.cross(a, b)
    s = xp.linalg.vector_norm(v)
    if not (s > 0):
        # a and b are colinear
        x, y, z = a
        if x > 0:
            v = xp.asarray([y, -x, 0])
        elif y > 0:
            v = xp.asarray([y, -x, 0])
        else:  # x == y == 0, a is colinear to z-axis
            v = xp.asarray([0, 0, 1])
        s = xp.linalg.vector_norm(v)
    c = xp.linalg.vecdot(a, b)
    ssc = skew_symmetric_cross_product(v)
    rot = xp.eye(3, 3, dtype=dtype, device=device) + ssc + (ssc @ ssc) * (1 - c) / s**2
    return rot


def threshold_otsu(
    image: "Array | None " = None,
    nbins: int | None = 256,
    *,
    hist: "Array | None" = None,
) -> "Array":
    xp = array_namespace(image)
    if is_cupy_array(image):
        from cucim.skimage.filters import threshold_otsu as threshold_otsu_cucim

        func = threshold_otsu_cucim
    elif is_numpy_array(image):
        from skimage.filters import threshold_otsu as threshold_otsu_skimage

        func = threshold_otsu_skimage
    else:
        raise NotImplementedError(f"threshold_otsu is not implemented for {xp} backend")

    return func(image, nbins, hist=hist)


def find_pose_from_z_axis_to_centriole_axis2(
    centriole_im: "Array", axis_indice: int = 0, convention="XZX"
):
    """New method to get the centriole axis
    We test all possible poses in SO3. We evaluate if the pose
    is good by projecting the volume rotated on the Z axis, thresholding it
    and looking at the total surface.
    If the surface is small we probably have the centriole seen from the top,
    if not the pose is probably wrong.
    """
    xp = array_namespace(centriole_im)
    device = get_device(centriole_im)
    M = 1000
    potential_poses, precision = create_poses_grid(
        xp, M, 1, device=device, dtype=xp.float64
    )

    H = get_transform_matrix_from_pose(
        centriole_im.shape,
        potential_poses,
        convention=convention,
    )

    volume_rotated = affine_transform(
        xp.stack((centriole_im,) * M),
        H,
        order=1,
        prefilter=False,
        batch=True,
        multichannel=False,
    )
    volume_rotated_proj = xp.sum(volume_rotated, axis=1 + axis_indice)
    thresh = xp.stack(
        [
            threshold_otsu(volume_rotated_proj[i])
            for i in range(volume_rotated_proj.shape[0])
        ]
    )[:, None, None]

    best = xp.argmin(
        xp.sum((volume_rotated_proj > thresh), axis=(1, 2)),
    )
    return potential_poses[best]


def find_centriole_symmetry_axis(
    centriole_im: "Array", threshold: float = DEFAULT_THRESHOLD
):
    xp = array_namespace(centriole_im)
    device = get_device(centriole_im)
    ma = xp.max(centriole_im)
    centriol_pc = convert_im_to_point_cloud(centriole_im, ma * threshold)

    # PCA in numpy
    pca = PCA(n_components=3)
    pca.fit(to_numpy(centriol_pc))
    sum_of_2_by_2_differences = np.zeros(3)
    for i in range(3):
        for j in range(3):
            if j != i:
                sum_of_2_by_2_differences[i] += np.abs(
                    pca.singular_values_[i] - pca.singular_values_[j]
                )
    idx_dim_pca = np.argmax(sum_of_2_by_2_differences)
    logger.debug(f"{pca.singular_values_=}")
    logger.debug(f"{pca.components_=}")
    logger.debug(f"{sum_of_2_by_2_differences=}")
    if logger.isEnabledFor(logging.DEBUG):
        import plotly.graph_objects as go

        from spfluo.utils.pointcloud_viz import interactive_plot

        center = np.mean(centriol_pc, axis=0)
        fig = interactive_plot([centriol_pc], return_fig=True)
        for i, comp in enumerate(pca.components_):
            fig.add_trace(
                go.Scatter3d(
                    x=[center[0], center[0] + comp[0]],
                    y=[center[1], center[1] + comp[1]],
                    z=[center[2], center[2] + comp[2]],
                    mode="lines",
                    line=dict(color="red", width=3),
                    name=f"Composante {i+1}",
                )
            )
        fig.show()

    # return a generic Array
    return xp.asarray(
        pca.components_[idx_dim_pca], device=device, dtype=xp.float64
    ), xp.asarray(pca.mean_, device=device, dtype=xp.float64)


def find_pose_from_z_axis_to_centriole_axis(
    centriole_im: "Array", axis_indice=0, threshold=DEFAULT_THRESHOLD, convention="XZX"
):
    """Find the pose of the transformation from the axis to the centriole"""
    xp = array_namespace(centriole_im)
    device = get_device(centriole_im)
    symmetry_axis, center = find_centriole_symmetry_axis(
        centriole_im, threshold=threshold
    )
    z_axis = xp.asarray([0, 0, 0], device=device, dtype=xp.float64)
    z_axis[axis_indice] = 1.0
    rot = find_rotation_between_two_vectors(symmetry_axis, z_axis)
    pose = xp.zeros((6,), dtype=xp.float64)
    pose[:3] = matrix_to_euler(convention, rot.T, degrees=True)
    return xp.asarray(pose, dtype=xp.float64, device=device)


def _gradient(im: "Array", nb_spatial_dims: int = 2):
    xp = array_namespace(im)
    grad = []
    for i in range(-nb_spatial_dims, 0):
        selection_a, selection_b = [], []
        for j in range(-im.ndim, 0):
            if j < -nb_spatial_dims:
                selection_a.append(slice(None))
                selection_b.append(slice(None))
            elif j == i:
                selection_a.append(slice(1, None))
                selection_b.append(slice(None, -1))
            else:
                selection_a.append(slice(None, -1))
                selection_b.append(slice(None, -1))
        grad_i = im[tuple(selection_a)] - im[tuple(selection_b)]
        grad.append(grad_i)
    return xp.stack(grad, axis=0)


def _corr(im1: "Array", im2: "Array", nb_spatial_dims: int = 2):
    xp = array_namespace(im1, im2)
    assert im1.shape[-nb_spatial_dims:] == im2.shape[-nb_spatial_dims:]
    mean = partial(xp.mean, axis=tuple(range(-nb_spatial_dims, 0)), keepdims=True)
    im1_stdv = mean((im1 - mean(im1)) ** 2) ** 0.5
    im2_stdv = mean((im2 - mean(im2)) ** 2) ** 0.5
    c = mean((im1 - mean(im1)) * (im2 - mean(im2))) / (im1_stdv * im2_stdv)
    c = c[
        (slice(None),) * (c.ndim - nb_spatial_dims) + (0,) * nb_spatial_dims
    ]  # squeeze dims
    return c


def find_pose_from_centriole_to_center(
    im: "Array", symmetry: int, precision: float = 1, axis_indice: int = 0
):
    xp = array_namespace(im)
    dtype = im.dtype
    device = get_device(im)
    # Z-trans
    axes = [0, 1, 2]
    axes.pop(axis_indice)
    axes = tuple(axes)
    center_mass = xp.sum(
        xp.arange(im.shape[axis_indice], device=get_device(im)) * xp.sum(im, axis=axes)
    ) / xp.sum(im)
    center = (float(im.shape[axis_indice]) - 1) / 2
    trans_z = center - center_mass

    # Y-X trans
    num = math.ceil(max(im.shape[1:]) / (4 * precision))
    N_trans = num * num
    im_proj = xp.sum(im, axis=axis_indice)
    yx_translations = xp.reshape(
        xp.stack(
            xp.meshgrid(xp.linspace(-1, 1, num), xp.linspace(-1, 1, num)), axis=-1
        ),
        (num * num, 2),
    )
    yx_translations = yx_translations * xp.asarray(im_proj.shape) / 4

    H_trans = xp.stack((xp.eye(3),) * N_trans)
    H_trans[:, :2, 2] = yx_translations

    angles = 2 * xp.arange(symmetry, dtype=dtype) * xp.pi / symmetry
    R = xp.permute_dims(
        xp.asarray(
            xp.stack(
                [
                    xp.stack([xp.cos(angles), -xp.sin(angles)]),
                    xp.stack([xp.sin(angles), xp.cos(angles)]),
                ]
            )
        ),
        axes=(2, 0, 1),
    )
    H_rot = get_transform_matrix_around_center(im_proj.shape, R)

    H = H_rot[None] @ H_trans[:, None]  # (N_trans, symmetry, 3, 3)
    H_inv = xp.linalg.inv(H)

    H_inv[:, :, 2, [0, 1, 2]] = xp.asarray(
        [0.0, 0.0, 1.0], dtype=H_inv.dtype, device=get_device(H_inv)
    )
    ims_translated_rotated = xp.reshape(
        xp.asarray(
            affine_transform(
                to_numpy(
                    xp.stack((im_proj,) * N_trans * symmetry)
                ),  # to numpy because affine transform only works in numpy for 2D
                to_numpy(xp.reshape(H_inv, (-1, 3, 3))),
                batch=True,
            ),
            device=device,
            dtype=dtype,
        ),
        (N_trans, symmetry, *im_proj.shape),
    )

    corr_rotations = xp.mean(
        _corr(
            xp.mean(ims_translated_rotated, axis=1, keepdims=True),
            ims_translated_rotated,
        ),
        axis=1,
    )

    grad = _gradient(ims_translated_rotated)
    corr_rotations_grads = xp.mean(
        _corr(xp.mean(grad, axis=2, keepdims=True), grad), axis=(0, 2)
    )

    distances = (corr_rotations + corr_rotations_grads) / 2

    trans_y, trans_x = yx_translations[xp.argmax(distances)]

    pose = xp.asarray([0, 0, 0, 0, 0, 0], dtype=xp.float64, device=device)
    pose[3 + axis_indice] = trans_z
    pose[3 + axes[0]] = trans_y
    pose[3 + axes[1]] = trans_x

    return pose


def find_pose_from_z_axis_centered_to_centriole_axis(
    centriole_im: "Array",
    symmetry: int,
    axis_indice=0,
    threshold: float = DEFAULT_THRESHOLD,
    center_precision: float = 1,
    convention="XZX",
):
    logger.info("start find_pose_from_z_axis_centered_to_centriole_axis")
    pose_from_z_axis_to_centriole = find_pose_from_z_axis_to_centriole_axis(
        centriole_im,
        axis_indice=axis_indice,
        threshold=threshold,
        convention=convention,
    )
    logger.debug(f"{pose_from_z_axis_to_centriole=}")
    volume_z_axis = affine_transform(
        centriole_im,
        get_transform_matrix_from_pose(
            centriole_im.shape, pose_from_z_axis_to_centriole, convention=convention
        ),
        order=1,
    )
    pose_from_z_axis_to_z_axis_centered = find_pose_from_centriole_to_center(
        volume_z_axis, symmetry, axis_indice=axis_indice, precision=center_precision
    )
    logger.debug(f"{pose_from_z_axis_to_z_axis_centered=}")
    return compose_poses(
        invert_pose(pose_from_z_axis_to_z_axis_centered), pose_from_z_axis_to_centriole
    )


def main(
    volume_path: str,
    symmetry: int,
    convention: str = "XZX",
    threshold: float = DEFAULT_THRESHOLD,
    output_volume_path: Optional[str] = None,
    poses_path: Optional[str] = None,
    output_poses_path: Optional[str] = None,
):
    tif = tifffile.TiffFile(volume_path, is_ome=True)
    volume = get_data_from_ome_tiff(tif, 0, order="CZYX")

    # Volume is mono-channel
    assert volume.shape[0] == 1, f"volume of shape {volume.shape} is not monochannel"
    volume = volume[0]

    ome = from_xml(tif.ome_metadata)
    assert len(ome.images) == 1

    # if volume has pixel sizes, resample it to be isotropic
    if (
        ome.images[0].pixels.physical_size_x
        and ome.images[0].pixels.physical_size_y
        and ome.images[0].pixels.physical_size_z
    ):
        # assert all physical sizes have the same unit
        assert (
            len(
                set(
                    [
                        ome.images[0].pixels.physical_size_x_unit,
                        ome.images[0].pixels.physical_size_y_unit,
                        ome.images[0].pixels.physical_size_z_unit,
                    ]
                )
            )
            == 1
        )
        target_pixel_physical_unit = ome.images[0].pixels.physical_size_x_unit
        target_pixel_physical_size = min(
            ome.images[0].pixels.physical_size_z,
            ome.images[0].pixels.physical_size_y,
            ome.images[0].pixels.physical_size_x,
        )

        volume = resample(
            volume,
            (
                ome.images[0].pixels.physical_size_z / target_pixel_physical_size,
                ome.images[0].pixels.physical_size_y / target_pixel_physical_size,
                ome.images[0].pixels.physical_size_x / target_pixel_physical_size,
            ),
        )
    else:
        target_pixel_physical_size = 1.0
        target_pixel_physical_unit = UnitsLength.MICROMETER

    # Volume must be a cube
    size = max(volume.shape)
    volume = interpolate_to_size(volume, (size, size, size))

    pose = find_pose_from_z_axis_centered_to_centriole_axis(
        volume,
        symmetry,
        threshold=threshold,
    )

    if output_volume_path:
        rotated_volume = affine_transform(
            volume,
            get_transform_matrix_from_pose(volume.shape, pose, convention=convention),
        )
        assert output_volume_path.endswith(".ome.tiff")
        tifffile.imwrite(
            output_volume_path,
            rotated_volume,
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": target_pixel_physical_size,
                "PhysicalSizeXUnit": target_pixel_physical_unit.value,
                "PhysicalSizeY": target_pixel_physical_size,
                "PhysicalSizeYUnit": target_pixel_physical_unit.value,
                "PhysicalSizeZ": target_pixel_physical_size,
                "PhysicalSizeZUnit": target_pixel_physical_unit.value,
            },
        )
    if poses_path:
        poses, names = read_poses(poses_path)
        new_poses = compose_poses(pose, poses, convention=convention)
        save_poses(output_poses_path, new_poses, names)
