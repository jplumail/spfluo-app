from spfluo.visualisation import add_orthoviewer_widget, init_qt

import napari
from napari.layers import Points
import numpy as np


def show_points(im_path, csv_path):
    init_qt()
    view = napari.Viewer()
    #view.window._qt_viewer.dockLayerList.toggleViewAction().trigger()
    view, dock_widget, cross = add_orthoviewer_widget(view)

    view.open(im_path, plugin='napari-aicsimageio')

    cross.setChecked(True)
    cross.hide()

    points_layer = Points(
        ndim=3,
        edge_color=[0,0,255,255],
        face_color=[0,0,0,0],
        out_of_slice_display=True,
        size=10
    )

    coords = []
    sizes = []
    with open(csv_path, mode='r') as csv:
        csv.readline()
        for line in csv:
            line = line.strip().split(',')
            coords.append([float(line[1]), float(line[2]), float(line[3])])
            sizes.append([float(line[4]), float(line[4]), float(line[4])])

    points_layer.data = np.array(coords)
    points_layer.size = np.array(sizes)

    view.add_layer(points_layer)

    napari.run()