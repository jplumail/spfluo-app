import tempfile
from typing import List

import napari
import numpy as np
import pandas as pd
from aicsimageio.aics_image import AICSImage
from napari.experimental import link_layers, unlink_layers
from napari.layers import Points

from spfluo.visualisation.multiple_viewer_widget import add_orthoviewer_widget, init_qt


def show_points(im_path: str, csv_path: str):
    init_qt()
    view = napari.Viewer()
    # view.window._qt_viewer.dockLayerList.toggleViewAction().trigger()
    view, dock_widget, cross = add_orthoviewer_widget(view)

    view.open(im_path, plugin="napari-aicsimageio")

    cross.setChecked(True)
    cross.hide()

    points_layer = Points(
        ndim=3,
        edge_color=[0, 0, 255, 255],
        face_color=[0, 0, 0, 0],
        out_of_slice_display=True,
        size=10,
    )

    coords = []
    sizes = []
    with open(csv_path, mode="r") as csv:
        csv.readline()
        for line in csv:
            line = line.strip().split(",")
            coords.append([float(line[1]), float(line[2]), float(line[3])])
            sizes.append([float(line[4]), float(line[4]), float(line[4])])

    points_layer.data = np.array(coords)
    points_layer.size = np.array(sizes)

    view.add_layer(points_layer)

    napari.run()


def show_particles(im_paths: List[str]):
    viewer = napari.Viewer()

    def indexer(p):
        return pd.Series(
            [im_paths.index(p), 0, 0, 0], index=["C", "S", "T", "Z"]
        ).astype(int)

    with tempfile.NamedTemporaryFile(suffix=".tif") as f:
        AICSImage(im_paths, indexer=indexer, single_file_dims=("Z", "Y", "X")).save(
            f.name
        )
        viewer.open(
            f.name, colormap="gray", name="particle", plugin="napari-aicsimageio"
        )
    link_layers(viewer.layers)
    unlink_layers(viewer.layers, attributes=("visible",))
    viewer.grid.enabled = True
    napari.run()
