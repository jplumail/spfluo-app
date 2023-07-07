import os
import pickle
import random
from typing import Dict, Tuple

import imageio
import numpy as np
import tifffile
import torch

try:
    import cupy as cp
    from cupy._core import ndarray
    from cupyx.scipy import ndimage

    use_cupy = True
except ImportError:
    from scipy import ndimage

    use_cupy = False
from spfluo.ab_initio_reconstruction.common_image_processing_methods.others import (
    crop_center,
)


def get_cupy_array(image):
    if use_cupy:
        return cp.array(image)
    else:
        return image


def get_numpy_array(image):
    if isinstance(image, ndarray):
        return image.get()
    else:
        return image


def seed_all(seed_numpy: bool = True) -> None:
    """For reproductibility.

    But, if we shuffle annotations between runs, we want each shuffling to be unique.
    Hence the option to choose if numpy must be seeded.
    """
    random.seed(0)
    torch.manual_seed(0)
    if seed_numpy:
        np.random.seed(0)


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


def get_spacing(im_path: str) -> Tuple[Tuple[float], str]:
    """
    Returns spacing (inverse of resolution) in x,y,z directions
    and unit (usually microns)
    """
    tif = tifffile.TiffFile(im_path)
    if tif.is_imagej:
        meta = tif.imagej_metadata
        res = tif.pages[0].resolution
        xy_spacing = (1 / res[0], 1 / res[1])
        z_spacing = float(meta["spacing"])
        spacing = xy_spacing + (z_spacing,)
        unit = meta["unit"]
    elif tif.is_ome:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(tif.ome_metadata)
        metadata = root[1][1].attrib
        spacing = (
            float(metadata["PhysicalSizeX"]),
            float(metadata["PhysicalSizeY"]),
            float(metadata["PhysicalSizeZ"]),
        )
        unit = metadata["PhysicalSizeZUnit"]
    else:
        return None, None
    return spacing, unit


def isotropic_resample(
    im_paths: str, folder_path: str, spacing: Tuple[float, float, float] = None
) -> None:
    if spacing is None:
        spacings = np.array([get_spacing(p)[0] for p in im_paths], dtype=float)
    else:
        spacings = np.ones((len(im_paths), 3))
        spacings[:] = np.array(spacing)
    best_spacing = spacings.min()
    zooms = spacings / best_spacing
    os.makedirs(folder_path, exist_ok=True)
    for im_path, zoom in zip(im_paths, zooms):
        im = tifffile.imread(im_path)
        im = get_cupy_array(im)
        ndimage.zoom(im, zoom[::-1], output=im)
        im = get_numpy_array(im)
        tifffile.imwrite(os.path.join(folder_path, os.path.basename(im_path)), im)


def resize(im_paths: str, size: int, folder_path: str):
    os.makedirs(folder_path, exist_ok=True)
    for im_path in im_paths:
        im = tifffile.imread(im_path)
        im_resized = crop_center(im, (size,) * 3)
        tifffile.imwrite(
            os.path.join(folder_path, os.path.basename(im_path)), im_resized
        )


def resample(im_paths: str, folder_path: str, factor: float = 1.0) -> None:
    os.makedirs(folder_path, exist_ok=True)
    for im_path in im_paths:
        im = tifffile.imread(im_path)
        im = get_cupy_array(im)
        im_resampled = ndimage.zoom(im, factor)
        im_resampled = get_numpy_array(im_resampled)
        tifffile.imwrite(
            os.path.join(folder_path, os.path.basename(im_path)), im_resampled
        )
