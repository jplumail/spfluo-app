import os
from typing import TYPE_CHECKING

import imageio
import numpy as np
import pandas as pd
import tifffile

from spfluo.utils.array import Array, array_namespace, get_namespace_device, numpy

if TYPE_CHECKING:
    from spfluo.utils.array import Device, array_api_module


def read_image(
    path,
    dtype: str | None = None,
    xp: "array_api_module | None" = None,  # type: ignore
    device: "Device | None" = None,
    gpu: bool | None = None,
) -> Array:
    xp, device = get_namespace_device(xp, device, gpu)
    arr = numpy.asarray(
        imageio.mimread(path, memtest=False), dtype=getattr(numpy, dtype)
    )
    return xp.asarray(arr, device=device)


def read_images_in_folder(
    folder,
    alphabetic_order=True,
    dtype: str | None = None,
    xp: "array_api_module | None" = None,  # type: ignore
    device: "Device | None" = None,
    gpu: bool | None = None,
) -> Array:
    """read all the images inside folder fold"""
    files = os.listdir(folder)
    if alphabetic_order:
        files = sorted(files)
    images = []
    for fn in files:
        pth = f"{folder}/{fn}"
        im = read_image(pth, dtype, xp, device, gpu)
        images.append(im)
    xp = array_namespace(im)
    return xp.stack(images), files


def save(path: str, array: Array):
    # save with conversion to float32 so that imaej can open it
    tifffile.imwrite(path, np.asarray(array, dtype=np.float32))


def make_dir(dir):
    """creates folder at location dir if i doesn't already exist"""
    if not os.path.exists(dir):
        print(f"directory {dir} created")
        os.makedirs(dir)


def write_array_csv(np_array, path):
    pd.DataFrame(np_array).to_csv(path)
