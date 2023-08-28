import logging
import pathlib
from datetime import datetime
from typing import Any

import numpy as np
import tifffile

DEBUG_DIR = pathlib.Path("spfluo_debug")
DEBUG_DIR_REFINEMENT = DEBUG_DIR / "refinement"


def create_debug_directories():
    global DEBUG_DIR, DEBUG_DIR_REFINEMENT
    if logging.getLogger("spfluo").isEnabledFor(logging.DEBUG):
        DEBUG_DIR.mkdir(exist_ok=True)
    if logging.getLogger("spfluo.refinement").isEnabledFor(logging.DEBUG):
        DEBUG_DIR_REFINEMENT.mkdir(parents=True, exist_ok=True)


def save_image(
    image: np.ndarray, directory: pathlib.Path, func: Any, *args: str, sequence=False
) -> str:
    create_debug_directories()
    ts = f"{datetime.now().timestamp():.3f}"
    names = "_".join(args)
    path = str(directory / (ts + "_" + func.__name__ + "_" + names)) + ".tiff"
    if sequence:
        metadata = {"axes": "TZYX"}
        tifffile.imwrite(path, image, metadata=metadata, imagej=True)
    tifffile.imwrite(path, image)
    return path
