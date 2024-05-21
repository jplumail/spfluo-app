import csv
import math
import os
import pickle
from typing import Dict, Optional, Tuple

import imageio
import numpy as np
import tifffile
from ome_types import OME, from_xml
from ome_types.model.simple_types import UnitsLength

import spfluo
from spfluo.utils.volume import interpolate_to_size
from spfluo.utils.volume import resample as _resample


def _get_dims_from_ome(ome: OME, scene_index: int) -> list[str]:
    """
    Process the OME metadata to retrieve the dimension names.

    Parameters
    ----------
    ome: OME
        A constructed OME object to retrieve data from.
    scene_index: int
        The current operating scene index to pull metadata from.

    Returns
    -------
    dims: List[str]
        The dimension names pulled from the OME metadata.

    Taken from aicsimageio
    """
    # Select scene
    scene_meta = ome.images[scene_index]

    # Create dimension order by getting the current scene's dimension order
    # and reversing it because OME store order vs use order is :shrug:
    dims = [d for d in scene_meta.pixels.dimension_order.value[::-1]]

    # Check for num samples and expand dims if greater than 1
    n_samples = scene_meta.pixels.channels[0].samples_per_pixel
    if n_samples is not None and n_samples > 1 and "S" not in dims:
        # Append to the end, i.e. the last dimension
        dims.append("S")

    return dims


def _guess_ome_dim_order(
    tiff: tifffile.TiffFile, ome: OME, scene_index: int
) -> list[str]:
    """
    Guess the dimension order based on OME metadata and actual TIFF data.
    Parameters
    -------
    tiff: TiffFile
        A constructed TIFF object to retrieve data from.
    ome: OME
        A constructed OME object to retrieve data from.
    scene_index: int
        The current operating scene index to pull metadata from.
    Returns
    -------
    dims: List[str]
        Educated guess of the dimension order for the file

    Taken from aicsimageio
    """
    dims_from_ome = _get_dims_from_ome(ome, scene_index)

    # Assumes the dimensions coming from here are align semantically
    # with the dimensions specified in this package. Possible T dimension
    # is not equivalent to T dimension here. However, any dimensions
    # not also found in OME will be omitted.
    dims_from_tiff_axes = list(tiff.series[scene_index].axes)

    # Adjust the guess of what the dimensions are based on the combined
    # information from the tiff axes and the OME metadata.
    # Necessary since while OME metadata should be source of truth, it
    # does not provide enough data to guess which dimension is Samples
    # for RGB files
    dims = [dim for dim in dims_from_ome if dim not in dims_from_tiff_axes]
    dims += [dim for dim in dims_from_tiff_axes if dim in dims_from_ome]
    return dims


def get_data_from_ome_tiff(
    tiff: tifffile.TiffFile, scene_index: int, order: str = "CZYX"
):
    """returns data in the order asked"""
    assert tiff.is_ome
    dims = _guess_ome_dim_order(tiff, from_xml(tiff.ome_metadata), scene_index)
    image_data = tiff.series[scene_index].asarray(squeeze=False)
    assert image_data.shape[5] == 1, f"{image_data.shape=}: S != 1"
    image_data = image_data[:, :, :, :, :, 0]
    unspecified_dims = set(dims).difference(set(order))
    for dim in unspecified_dims:
        assert image_data.shape[dims.index(dim)] == 1
    image_data = image_data.squeeze(
        axis=tuple([dims.index(dim) for dim in unspecified_dims])
    )
    dims = [dim for dim in dims if dim not in unspecified_dims]
    assert len(order) == len(dims), f"{len(order)=} != {len(dims)=}"
    return image_data.transpose(*tuple([dims.index(o) for o in order]))


def get_cupy_array(image):
    if spfluo.has_cupy():
        import cupy as cp

        return cp.array(image)
    else:
        return image


def get_ndimage():
    if spfluo.has_cupy():
        from cupyx.scipy import ndimage
    else:
        from scipy import ndimage
    return ndimage


def get_numpy_array(image):
    if not isinstance(image, np.ndarray):
        return image.get()
    else:
        return image


def load_array(path: str) -> np.ndarray:
    """Takes a complete path to a file and return the corresponding numpy array.

    Args:
        path (str): Path to the file to read. It MUST contains the extension as this
                    will determines the way the numpy array is loaded from the file.

    Returns:
        np.ndarray: The array stored in the file described by 'path'.
    """
    extension = os.path.splitext(path)[-1]
    if extension == ".npz":
        return np.load(path)["image"]
    elif extension in [".tif", ".tiff"]:
        image = imageio.volread(path).astype(np.int16)
        # Some tiff images are heavily imbalanced:
        # their data type is int16 but very few voxels are actually > 255.
        # If this is the case, the image in truncated and casted to uint8.
        if image.dtype == np.int16 and ((image > 255).sum() / image.size) < 1e-3:
            image[image > 255] = 255
            image = image.astype(np.uint8)
        return image
    error_msg = f"Found extension {extension}. Extension must be one of npz or tif."
    raise NotImplementedError(error_msg)


def load_annotations(csv_path: str) -> np.ndarray:
    """Csv containing coordinates of objects center.

    Args:
        csv_path (str): Path of the file to read.

    Returns:
        np.ndarray: Array into which each line is alike
            ('image name', particle_id, z, y, x).
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line_data = line.split(",")
        line_data[2:] = list(map(float, line_data[2:]))
        data.append(line_data)
    return np.array(data, dtype=object)


def load_angles(csv_path: str) -> np.ndarray:
    """Csv containing angles of particles.

    Args:
        csv_path (str): Path of the file to read.

    Returns:
        np.ndarray: Array into which each line is alike
            ('image name', particle_id, z, y, x).
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line_data = line.split(",")
        line_data[2:] = list(map(float, line_data[2:]))
        data.append(line_data)
    return np.array(data, dtype=object)


def load_views(views_path, extension=None):
    _, ext = os.path.splitext(views_path)
    if ext == ".csv":  # DONT USE CSV NOT WORKING
        views_ = load_annotations(views_path)
        views = views_[:, 2].astype(int)
        if extension is None:
            patches_names = np.array(
                [
                    os.path.splitext(im_name)[0]
                    + "_"
                    + patch_index
                    + os.path.splitext(im_name)[1]
                    for im_name, patch_index in views_[:, [0, 1]]
                ]
            )
        else:
            patches_names = np.array(
                [
                    os.path.splitext(im_name)[0] + "_" + patch_index + extension
                    for im_name, patch_index in views_[:, [0, 1]]
                ]
            )

    elif ext == ".pickle":
        with open(views_path, "rb") as f:
            views_ = pickle.load(f)
        views = np.concatenate([views_[image_name][0] for image_name in views_.keys()])
        patches_names = np.concatenate(
            [views_[image_name][2] for image_name in views_.keys()]
        )

    return views, patches_names


def load_patches(crop_dir, patches_names):
    patches = [
        load_array(os.path.join(crop_dir, p)) for p in patches_names
    ]  # (N,z,y,x)
    return np.stack(patches, axis=0)


def load_pointcloud(pointcloud_path: str) -> np.ndarray:
    template_point_cloud = np.loadtxt(pointcloud_path, delimiter=",")
    return template_point_cloud


def center_to_corners(center: Tuple[int], size: Tuple[int]) -> Tuple[int]:
    depth, height, width = size
    center_z, center_y, center_x = center
    z_min = center_z - depth // 2
    y_min = center_y - height // 2
    x_min = center_x - width // 2
    z_max = z_min + depth
    y_max = y_min + height
    x_max = x_min + width
    return x_min, y_min, z_min, x_max, y_max, z_max


def summary(
    kwargs: Dict, title: str, output: str = None, return_table: bool = False
) -> None:
    # 1. Get key max length
    key_length = max([len(k) for k in kwargs.keys()])
    # 2. Get total length
    length = max([key_length + len(str(v)) for k, v in kwargs.items()]) + 4
    # 3. Define table delimiter and title
    hbar = "+" + length * "-" + "+" + "\n"
    pad_left = (length - len(title)) // 2
    pad_right = length - len(title) - pad_left
    title = "|" + pad_left * " " + title + pad_right * " " + "|" + "\n"
    # 4. Create table string
    table = "\n" + hbar + title + hbar
    # 5. Add lines to table
    for k, v in kwargs.items():
        line = k + (key_length - len(k)) * "." + ": " + str(v)
        line = "| " + line + (length - 1 - len(line)) * " " + "|" + "\n"
        table += line
    table += hbar
    # 6. Print table
    print(table)
    # 7. Save table to txt file (optional)
    if output is not None:
        with open(output, "w") as file:
            file.write(table)
    # 8. Return table (optional)
    if return_table:
        return table


def send_mail(subject: str, body: str) -> None:
    import smtplib
    import ssl

    port = 465  # SSL
    password = "hackitifyouwantidontcare"
    sender_email = "icuberemotedev@gmail.com"
    receiver_email = "vedrenneluc@gmail.com"
    message = f"""\
    {subject}

    {body}
    """
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


def crop_one_particle(
    image: np.ndarray,
    center: np.ndarray,
    crop_size: Tuple[int],
    max_size: Tuple[int],
) -> np.ndarray:
    corners = center_to_corners(center, crop_size)
    corners = reframe_corners_if_needed(corners, crop_size, max_size)
    x_min, y_min, z_min, x_max, y_max, z_max = corners
    return image[z_min:z_max, y_min:y_max, x_min:x_max]


def reframe_corners_if_needed(
    corners: Tuple[int], crop_size: Tuple[int], max_size: Tuple[int]
) -> Tuple[int]:
    d, h, w = crop_size
    D, H, W = max_size
    x_min, y_min, z_min, x_max, y_max, z_max = corners
    z_min, y_min, x_min = max(0, z_min), max(0, y_min), max(0, x_min)
    # z_max, y_max, x_max = z_min + d, y_min + h, x_min + w
    z_max, y_max, x_max = min(D, z_max), min(H, y_max), min(W, x_max)
    # z_min, y_min, x_min = z_max - d, y_max - h, x_max - w
    return x_min, y_min, z_min, x_max, y_max, z_max


def resize(im_paths: str, size: float, folder_path: str):
    """pad isotropic OME-TIFF images to match the shape of a cube of size (size, size,
    size).
    If size isn't a multiple of the pixel size, new_size = ceil(size/pixel_size)
    Arguments:
        im_paths
        size
            shape of the cube in µm
        folder_path
            output folder
    """
    target_physical_size = size
    os.makedirs(folder_path, exist_ok=True)
    for im_path in im_paths:
        tif = tifffile.TiffFile(im_path, is_ome=True)
        ome = from_xml(tif.ome_metadata)
        im = get_data_from_ome_tiff(tif, 0, order="CZYX")

        assert len(ome.images) == 1
        # assert image is isotropic
        (pixel_physical_size,) = set(
            [
                ome.images[0].pixels.physical_size_x,
                ome.images[0].pixels.physical_size_y,
                ome.images[0].pixels.physical_size_z,
            ]
        )
        (pixel_physical_size_unit,) = set(
            [
                ome.images[0].pixels.physical_size_x_unit,
                ome.images[0].pixels.physical_size_y_unit,
                ome.images[0].pixels.physical_size_z_unit,
            ]
        )

        assert pixel_physical_size_unit == UnitsLength.MICROMETER
        pixel_size = math.ceil(target_physical_size / pixel_physical_size)
        im_resized = interpolate_to_size(im, (pixel_size,) * 3, multichannel=True)
        filename = os.path.join(folder_path, os.path.basename(im_path))

        # copy ome metadata to filename
        tifffile.imwrite(
            filename,
            im_resized,
            metadata={
                "axes": "CZYX",
                "PhysicalSizeX": pixel_physical_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_physical_size,
                "PhysicalSizeYUnit": "µm",
                "PhysicalSizeZ": pixel_physical_size,
                "PhysicalSizeZUnit": "µm",
            },
        )
        tif.close()


def resample(im_paths: str, folder_path: str, target_pixel_physical_size: float = 1.0):
    """resample images to match the target physical size
    Args
        target_physical_size: float
            in µm
    """
    os.makedirs(folder_path, exist_ok=True)
    for im_path in im_paths:
        tif = tifffile.TiffFile(im_path, is_ome=True)
        image = get_data_from_ome_tiff(tif, 0, order="CZYX")
        ome = from_xml(tif.ome_metadata)

        assert len(ome.images) == 1
        # assert image pixel units are µm
        (pixel_physical_size_unit,) = set(
            [
                ome.images[0].pixels.physical_size_x_unit,
                ome.images[0].pixels.physical_size_y_unit,
                ome.images[0].pixels.physical_size_z_unit,
            ]
        )
        assert (
            pixel_physical_size_unit == UnitsLength.MICROMETER
        ), f"Unit is different than µm, it's {pixel_physical_size_unit.value}"

        image_resampled = _resample(
            image,
            (
                ome.images[0].pixels.physical_size_z / target_pixel_physical_size,
                ome.images[0].pixels.physical_size_y / target_pixel_physical_size,
                ome.images[0].pixels.physical_size_x / target_pixel_physical_size,
            ),
            multichannel=True,
        )

        filename = os.path.join(folder_path, os.path.basename(im_path))
        tifffile.imwrite(
            filename,
            image_resampled,
            metadata={
                "axes": "CZYX",
                "PhysicalSizeX": target_pixel_physical_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": target_pixel_physical_size,
                "PhysicalSizeYUnit": "µm",
                "PhysicalSizeZ": target_pixel_physical_size,
                "PhysicalSizeZUnit": "µm",
            },
        )
        tif.close()


def save_poses(path: str, poses: np.ndarray, names: Optional[list[str]] = None):
    with open(path, "w") as f:
        f.write("name,rot1,rot2,rot3,t1,t2,t3\n")
        for i, p in enumerate(poses):
            pose = list(map(str, p.tolist()))
            name = names[i] if names else str(i)
            f.write(",".join([name] + pose))
            f.write("\n")


def read_poses(path: str, alphabetic_order=True):
    content = csv.reader(open(path, "r").read().split("\n"))
    next(content)
    poses, fnames = [], []
    for row in content:
        if len(row) > 0:
            poses.append(np.array(row[1:], dtype=float))
            fnames.append(row[0])
    if alphabetic_order:
        fnames, poses = zip(*sorted(zip(fnames, poses), key=lambda x: x[0]))
    poses = np.stack(poses)
    return poses, list(fnames)
