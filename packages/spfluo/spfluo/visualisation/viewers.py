import atexit
import os
import tempfile
from typing import List, Tuple

import napari
import numpy as np
import tifffile

from spfluo.manual_picking.annotate import annotate


def show_points(
    im_path: str, csv_path: str, scale: tuple[float, float, float] | None = None
):
    scale = scale if scale else (1, 1, 1)
    annotate(im_path, csv_path, spacing=scale, save=False)


def show_particles(im_paths: List[str], scale: Tuple[float, float, float] = None):
    viewer = napari.Viewer()

    f = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    f.close()
    atexit.register(lambda: os.remove(f.name))
    ims = np.stack([tifffile.imread(p) for p in im_paths])
    tifffile.imwrite(f.name, ims)
    (layer,) = viewer.open(f.name, colormap="gray", name="particle")
    if scale:
        layer.scale = scale
    napari.run()
