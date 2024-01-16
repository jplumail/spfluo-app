import atexit
import os
import tempfile
from typing import List

import napari
import numpy as np
import tifffile
from napari.experimental import link_layers, unlink_layers

from spfluo.manual_picking.annotate import annotate


def show_points(im_path: str, csv_path: str, scale: tuple[float, float, float] = None):
    annotate(im_path, csv_path, spacing=scale, save=False)


def show_particles(im_paths: List[str]):
    viewer = napari.Viewer()

    f = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    f.close()
    atexit.register(lambda: os.remove(f.name))
    ims = np.stack([tifffile.imread(p) for p in im_paths])
    tifffile.imwrite(f.name, ims)
    viewer.open(f.name, colormap="gray", name="particle")
    link_layers(viewer.layers)
    unlink_layers(viewer.layers, attributes=("visible",))
    viewer.grid.enabled = True
    napari.run()
