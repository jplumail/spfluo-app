import logging
import pathlib
from datetime import datetime
from typing import Any

import numpy as np
import tifffile

DEBUG_DIR = pathlib.Path("spfluo_debug")
DEBUG_DIR_REFINEMENT = DEBUG_DIR / "refinement"


def arg_pretty_repr(arg):
    if hasattr(arg, "shape"):
        return array_pretty_repr(arg)
    else:
        return repr(arg)


def array_pretty_repr(arr):
    return f"({type(arr)}, {arr.shape}, {arr.dtype})"


def create_debug_directories():
    global DEBUG_DIR, DEBUG_DIR_REFINEMENT
    if logging.getLogger("spfluo").isEnabledFor(logging.DEBUG):
        DEBUG_DIR.mkdir(exist_ok=True)
    if logging.getLogger("spfluo.refinement").isEnabledFor(logging.DEBUG):
        DEBUG_DIR_REFINEMENT.mkdir(parents=True, exist_ok=True)


def save_image(
    image: np.ndarray,
    directory: pathlib.Path,
    func: Any,
    *args: str,
    sequence=False,
    multichannel=False,
) -> str:
    create_debug_directories()
    ts = f"{datetime.now().timestamp():.3f}"
    names = "_".join(args)
    path = str(directory / (ts + "_" + func.__name__ + "_" + names)) + ".ome.tiff"
    axes = "ZYX"
    if multichannel:
        axes = "CZYX"
    if sequence:
        axes = "T" + axes
    assert image.ndim == len(axes), f"len({image.shape=} doesnt match {axes=}"
    tifffile.imwrite(path, image, metadata={"axes": axes})
    return path
