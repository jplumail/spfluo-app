import os
import pathlib
import tempfile
from datetime import datetime
from typing import Any

import numpy as np
import tifffile

DEBUG_DIR: pathlib.Path = None


def create_debug_directory() -> pathlib.Path:
    global DEBUG_DIR
    if DEBUG_DIR is None:
        DEBUG_DIR = pathlib.Path(
            tempfile.mkdtemp(prefix="spfluo_debug_", dir=os.getcwd())
        )


create_debug_directory()


def save_image(
    image: np.ndarray, directory: pathlib.Path, func: Any, *args: str, sequence=False
) -> str:
    ts = f"{datetime.now().timestamp():.3f}"
    names = "_".join(args)
    path = str(directory / (ts + "_" + func.__name__ + "_" + names)) + ".tiff"
    if sequence:
        metadata = {"axes": "TZYX"}
        tifffile.imwrite(path, image, metadata=metadata, imagej=True)
    tifffile.imwrite(path, image)
    return path
