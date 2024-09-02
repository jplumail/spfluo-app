# Some functions were taken from aicsimageio source code
from __future__ import annotations

import csv
import math
import os
import pickle
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import imageio
import numpy as np
import tifffile
from ome_types import OME, from_xml
from ome_types.model.simple_types import UnitsLength

import spfluo
from spfluo.utils.volume import interpolate_to_size
from spfluo.utils.volume import resample as _resample


class DimensionNames:
    Time = "T"
    Channel = "C"
    SpatialZ = "Z"
    SpatialY = "Y"
    SpatialX = "X"
    Samples = "S"
    MosaicTile = "M"


DEFAULT_DIMENSION_ORDER_LIST = [
    DimensionNames.Time,
    DimensionNames.Channel,
    DimensionNames.SpatialZ,
    DimensionNames.SpatialY,
    DimensionNames.SpatialX,
]

DEFAULT_DIMENSION_ORDER = "".join(DEFAULT_DIMENSION_ORDER_LIST)

DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES = DEFAULT_DIMENSION_ORDER_LIST + [
    DimensionNames.Samples
]

DEFAULT_DIMENSION_ORDER_WITH_SAMPLES = "".join(
    DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES
)


def transpose_to_dims(
    data: np.ndarray,
    given_dims: str,
    return_dims: str,
) -> np.ndarray:
    """
    This shuffles the data dimensions from given_dims to return_dims. Each dimension
    must be present in given_dims must be used in return_dims

    Parameters
    ----------
    data: types.ArrayLike
        Either a dask array or numpy.ndarray of arbitrary shape but with the dimensions
        specified in given_dims
    given_dims: str
        The dimension ordering of data, "CZYX", "VBTCXZY" etc
    return_dims: str
        The dimension ordering of the return data

    Returns
    -------
    data: types.ArrayLike
        The data with the specified dimension ordering.

    Raises
    ------
    ConflictingArgumentsError
        given_dims and return_dims are incompatible.
    """
    # Use a counter to track that the contents are composed of the same letters
    # and that no letter is repeated
    if (
        Counter(given_dims) != Counter(return_dims)
        or max(Counter(given_dims).values()) > 1
    ):
        raise ValueError(
            f"given_dims={given_dims} and return_dims={return_dims} are incompatible."
        )

    # Resort the data into return_dims order
    match_map = {dim: given_dims.find(dim) for dim in given_dims}
    transposer = []
    for dim in return_dims:
        transposer.append(match_map[dim])
    data = data.transpose(transposer)

    return data


def reduce_to_slice(L: list | tuple) -> int | list | slice | tuple:
    # if the list only has one element, then just use it
    if len(L) == 1:
        return slice(L[0], L[0] + 1)
    # if the list has at least 2 elements we can check for sliceable
    # it is convertable to a slice if the step size between each
    # consecutive pair of elements is equal and positive
    # 1. get all the deltas in a list:
    steps = [(L[i + 1] - L[i]) for i in range(len(L) - 1)]
    # 2. check if all the deltas are equal and positive
    if steps[0] > 0 and steps.count(steps[0]) == len(steps):
        return slice(min(L), max(L) + 1, steps[0])
    # if we can't convert to a slice, then just return the list unmodified
    return L


def reshape_data(
    data: np.ndarray, given_dims: str, return_dims: str, **kwargs: Any
) -> np.ndarray:
    """
    Reshape the data into return_dims, pad missing dimensions, and prune extra
    dimensions. Warns the user to use the base reader if the depth of the Dimension
    being removed is not 1.

    Parameters
    ----------
    data: types.ArrayLike
        Either a dask array or numpy.ndarray of arbitrary shape but with the dimensions
        specified in given_dims
    given_dims: str
        The dimension ordering of data, "CZYX", "VBTCXZY" etc
    return_dims: str
        The dimension ordering of the return data
    kwargs:
        * C=1 => desired specific channel, if C in the input data has depth 3 then C=1
          returns the 2nd slice (0 indexed)
        * Z=10 => desired specific channel, if Z in the input data has depth 20 then
          Z=10 returns the 11th slice
        * T=[0, 1] => desired specific timepoints, if T in the input data has depth 100
          then T=[0, 1] returns the 1st and 2nd slice (0 indexed)
        * T=(0, 1) => desired specific timepoints, if T in the input data has depth 100
          then T=(0, 1) returns the 1st and 2nd slice (0 indexed)
        * T=(0, -1) => desired specific timepoints, if T in the input data has depth 100
          then T=(0, -1) returns the first and last slice
        * T=range(10) => desired specific timepoints, if T in the input data has depth
          100 then T=range(10) returns the first ten slices
        * T=slice(0, -1, 5) => desired specific timepoints, T=slice(0, -1, 5) returns
          every fifth timepoint

    Returns
    -------
    data: types.ArrayLike
        The data with the specified dimension ordering.

    Raises
    ------
    ConflictingArgumentsError
        Missing dimension in return dims when using range, slice, or multi-index
        dimension selection for the requested dimension.

    IndexError
        Requested dimension index not present in data.

    Examples
    --------
    Specific index selection

    >>> data = np.random.rand((10, 100, 100))
    ... z1 = reshape_data(data, "ZYX", "YX", Z=1)

    List of index selection

    >>> data = np.random.rand((10, 100, 100))
    ... first_and_second = reshape_data(data, "ZYX", "YX", Z=[0, 1])

    Tuple of index selection

    >>> data = np.random.rand((10, 100, 100))
    ... first_and_last = reshape_data(data, "ZYX", "YX", Z=(0, -1))

    Range of index selection

    >>> data = np.random.rand((10, 100, 100))
    ... first_three = reshape_data(data, "ZYX", "YX", Z=range(3))

    Slice selection

    >>> data = np.random.rand((10, 100, 100))
    ... every_other = reshape_data(data, "ZYX", "YX", Z=slice(0, -1, 2))

    Empty dimension expansion

    >>> data = np.random.rand((10, 100, 100))
    ... with_time = reshape_data(data, "ZYX", "TZYX")

    Dimension order shuffle

    >>> data = np.random.rand((10, 100, 100))
    ... as_zx_base = reshape_data(data, "ZYX", "YZX")

    Selections, empty dimension expansions, and dimension order shuffle

    >>> data = np.random.rand((10, 100, 100))
    ... example = reshape_data(data, "CYX", "BSTCZYX", C=slice(0, -1, 3))
    """
    # Check for parameter conflicts
    for dim in given_dims:
        # return_dims='CZYX' and iterable dimensions 'T=range(10)'
        # Dimension is in kwargs
        # Dimension is an iterable
        # Dimension is not in return dimensions
        if (
            isinstance(kwargs.get(dim), (list, tuple, range, slice))
            and dim not in return_dims
        ):
            raise ValueError(
                f"When selecting a multiple dimension indices, the specified "
                f"dimension must be provided in return_dims. "
                f"return_dims={return_dims}, dimension {dim} = {kwargs.get(dim)}"
            )

    # Process each dimension available
    new_dims = given_dims
    dim_specs = []
    for dim in given_dims:
        # Store index of the dim as it is in given data
        dim_index = given_dims.index(dim)

        # Handle dim in return_dims which means that it is
        # an iterable or None selection
        if dim in return_dims:
            # Specific iterable requested
            if dim in kwargs:
                # Actual dim specification
                # The specification provided for this dimension in the kwargs
                dim_spec = kwargs.get(dim)
                display_dim_spec = dim_spec

                if isinstance(dim_spec, int):
                    dim_spec = slice(dim_spec, dim_spec + 1)

                # Convert operator to standard list or slice
                # dask.Array and numpy.ndarray both natively support
                # List[int] and slices being passed to getitem so no need to cast them
                # to anything different
                if isinstance(dim_spec, (tuple, range)):
                    dim_spec = list(dim_spec)

                # Get the largest absolute value index in the list using min and max
                if isinstance(dim_spec, list):
                    check_selection_max = max([abs(min(dim_spec)), max(dim_spec)])
                    # try to convert to slice if possible
                    dim_spec = reduce_to_slice(dim_spec)

                # Get the largest absolute value index from start and stop of slice
                if isinstance(dim_spec, slice):
                    check_selection_max = max([abs(dim_spec.stop), abs(dim_spec.start)])
            else:
                # Nothing was requested from this dimension
                dim_spec = slice(None, None, None)
                display_dim_spec = dim_spec

                # No op means that it doesn't matter how much data is in this dimension
                check_selection_max = 0

        # Not in return_dims means that it is a fixed integer selection
        else:
            if dim in kwargs:
                # Integer requested
                dim_spec = kwargs.get(dim)
                display_dim_spec = dim_spec

                # Check that integer
                check_selection_max = dim_spec
            else:
                dim_spec = 0
                display_dim_spec = dim_spec
                check_selection_max = 0

            # Remove dim from new dims as it is fixed size
            new_dims = new_dims.replace(dim, "")

        # Check that fixed integer request isn't outside of request
        if check_selection_max > data.shape[dim_index]:
            raise IndexError(
                f"Dimension specified with {dim}={display_dim_spec} "
                f"but Dimension shape is {data.shape[dim_index]}."
            )

        # All checks and operations passed, append dim operation to getitem ops
        dim_specs.append(dim_spec)

    # Run getitems
    data = data[tuple(dim_specs)]

    # Add empty dims where dimensions were requested but data doesn't exist
    # Add dimensions to new dims where empty dims are added
    for i, dim in enumerate(return_dims):
        # This dimension wasn't processed
        if dim not in given_dims:
            new_dims = f"{new_dims[:i]}{dim}{new_dims[i:]}"
            data = data.reshape(*data.shape[:i], 1, *data.shape[i:])

    # Any extra dimensions have been removed, only a problem if the depth is > 1
    return transpose_to_dims(
        data, given_dims=new_dims, return_dims=return_dims
    )  # don't pass kwargs or 2 copies


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


def _expand_dims_to_match_ome(
    image_data: np.ndarray,
    ome: OME,
    dims: list[str],
    scene_index: int,
) -> np.ndarray:
    # Expand image_data for empty dimensions
    ome_shape = []

    # need to correct channel count if this is a RGB image
    n_samples = ome.images[scene_index].pixels.channels[0].samples_per_pixel
    has_multiple_samples = n_samples is not None and n_samples > 1
    for d in dims:
        # SizeC can represent RGB (Samples) data rather
        # than channel data, whether or not this is the case depends
        # on what the SamplesPerPixel are for the channel
        if d == "C" and has_multiple_samples:
            count = len(ome.images[scene_index].pixels.channels)
        elif d == "S" and has_multiple_samples:
            count = n_samples
        else:
            count = getattr(ome.images[scene_index].pixels, f"size_{d.lower()}")
        ome_shape.append(count)

    # The file may not have all the data but OME requires certain dimensions
    # expand to fill
    expand_dim_ops: list[Optional[slice]] = []
    for d_size in ome_shape:
        # Add empty dimension where OME requires dimension but no data exists
        if d_size == 1:
            expand_dim_ops.append(None)
        # Add full slice where data exists
        else:
            expand_dim_ops.append(slice(None, None, None))

    # Apply operators to dask array
    return image_data[tuple(expand_dim_ops)]


def _general_data_array_constructor(
    image_data: np.ndarray,
    ome: OME,
    scene_index: int,
    dims: list[str],
):
    # Expand the image data to match the OME empty dimensions
    image_data = _expand_dims_to_match_ome(
        image_data=image_data,
        ome=ome,
        dims=dims,
        scene_index=scene_index,
    )

    # Always order array
    if DimensionNames.Samples in dims:
        out_order = DEFAULT_DIMENSION_ORDER_WITH_SAMPLES
    else:
        out_order = DEFAULT_DIMENSION_ORDER

    # Transform into order
    image_data = reshape_data(
        image_data,
        "".join(dims),
        out_order,
    )

    # Reset dims after transform
    dims = [d for d in out_order]

    return image_data, dims


def get_data_from_ome_tiff(
    tiff: tifffile.TiffFile, scene_index: int, order: str = "CZYX"
):
    """returns data in the order asked"""
    assert tiff.is_ome
    dims = _guess_ome_dim_order(tiff, from_xml(tiff.ome_metadata), scene_index)
    image_data, dims = _general_data_array_constructor(
        tiff.series[scene_index].asarray(),
        from_xml(tiff.ome_metadata),
        scene_index,
        dims,
    )

    # Special rule:
    # if T>1 and C=1, transpose
    if (
        "C" in order
        and "T" not in order
        and image_data.shape[dims.index("T")] > 1
        and image_data.shape[dims.index("C")] == 1
    ):
        x = list(range(image_data.ndim))
        x[dims.index("T")] = dims.index("C")
        x[dims.index("C")] = dims.index("T")
        image_data = image_data.transpose(*tuple(x))

    unspecified_dims = set(dims).difference(set(order))
    for dim in unspecified_dims:
        assert (
            image_data.shape[dims.index(dim)] == 1
        ), f"the dimension {dim} is not 1 but {image_data.shape[dims.index(dim)]} "
        " in image of shape {image_data.shape}"
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
