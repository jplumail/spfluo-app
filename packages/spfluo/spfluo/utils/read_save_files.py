from __future__ import annotations

import os
from typing import TYPE_CHECKING

import imageio
import numpy as np
import tifffile

from spfluo.utils.array import array_namespace, get_prefered_namespace_device, numpy

if TYPE_CHECKING:
    from spfluo.utils.array import Array, Device, array_api_module


def read_image(
    path,
    dtype: str | None = None,
    xp: "array_api_module | None" = None,  # type: ignore
    device: "Device | None" = None,
    gpu: bool | None = None,
):
    xp, device = get_prefered_namespace_device(xp, device, gpu)
    arr = numpy.asarray(
        imageio.mimread(path, memtest=False),
        dtype=getattr(numpy, dtype) if dtype else None,
    )
    return xp.asarray(arr, device=device)


def read_images_in_folder(
    folder,
    alphabetic_order=True,
    dtype: str | None = None,
    xp: "array_api_module | None" = None,  # type: ignore
    device: "Device | None" = None,
    gpu: bool | None = None,
) -> "Array":
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


def save_image(path: str, array: "Array", order: str = None, metadata: dict = None):
    assert path.endswith(".ome.tiff")
    metadata = {} if metadata is None else metadata
    if order is not None:
        metadata.update({"axes": order})
    tifffile.imwrite(path, np.asarray(array, dtype=np.float32), metadata=metadata)


def make_dir(dir):
    """creates folder at location dir if i doesn't already exist"""
    if not os.path.exists(dir):
        print(f"directory {dir} created")
        os.makedirs(dir)


def write_array_csv(
    arr: "Array", path: str, sep: str = ",", names: list[str] | None = None
):
    assert arr.ndim <= 2
    assert len(sep) == 1
    if arr.ndim == 1:
        arr = arr[:, None]
    with open(path, "w") as f:
        if names:
            assert len(names) == arr.shape[1]
            for name in names:
                f.write(name)
                f.write(sep)
            f.seek(f.tell() - 1)
            f.write("\n")
        for line in arr:
            for c in line:
                f.write(str(c))
                f.write(sep)
            f.seek(f.tell() - 1)
            f.write("\n")
