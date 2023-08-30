import csv
import os
from typing import Tuple

import napari
import numpy as np
from napari.layers import Points
from napari.utils.events import Event

from spfluo.visualisation.multiple_viewer_widget import add_orthoviewer_widget, init_qt


def annotate(
    im_path: str,
    output_path: str,
    size: int = 10,
    spacing: Tuple[float, float, float] = (1, 1, 1),
):
    """
    Inputs:
        im_path: the 3D image to read.
        output_path: the path of the output CSV. If a CSV is already present,
            read it and display.
        size: is in pixels (cannot do it in real size)
        spacing: is the size of the pixels in um, ZYX order.
    Ouput:
        The CSV written in output_path is in real coordinates, in um.
        The last column is the diameter of the circle in um.
    """
    init_qt()

    points_layer = Points(
        ndim=3,
        edge_color=[0, 0, 255, 255],
        face_color=[0, 0, 0, 0],
        out_of_slice_display=True,
        size=size,
        scale=spacing,
        name="Picking",
    )

    if os.path.exists(output_path):
        with open(output_path, "r", newline="") as csvfile:
            # read csv file into array
            data = []
            reader = csv.reader(csvfile)
            try:
                next(reader)
            except StopIteration:  # file is empty
                reader = []
            for p in reader:
                data.append(
                    points_layer.world_to_data((float(p[1]), float(p[2]), float(p[3])))
                )
                size = float(p[4])
            data = np.array(data)
        points_layer.data = data
        points_layer.current_size = int(size / spacing[1])

    view = napari.Viewer()
    view, dock_widget, cross = add_orthoviewer_widget(view)

    view.open(im_path, plugin="napari-aicsimageio", layer_type="image", scale=spacing)

    def on_move_point(event: Event):
        layer: Points = event.source
        viewers = [
            dock_widget.viewer,
            dock_widget.viewer_model1,
            dock_widget.viewer_model2,
        ]
        if len(layer.selected_data) > 0:
            idx_point = list(layer.selected_data)[0]
            try:
                pos_point = tuple(layer.data[idx_point])
            except IndexError:
                return

            # update viewers
            viewers_not_under_mouse = [
                viewer for viewer in viewers if not viewer.mouse_over_canvas
            ]
            if (
                len(viewers_not_under_mouse) == 3
            ):  # if mouse is not over any viewer, the point size is being adjusted
                return
            for viewer in viewers_not_under_mouse:
                pos_reordered = tuple(
                    np.array(points_layer.data_to_world(pos_point))[
                        list(viewer.dims.order)
                    ]
                )
                viewer.camera.center = pos_reordered

            dock_widget.viewer.dims.current_step = tuple(
                np.round(
                    [
                        max(min_, min(p, max_)) / step
                        for p, (min_, max_, step) in zip(
                            points_layer.data_to_world(pos_point),
                            dock_widget.viewer.dims.range,
                        )
                    ]
                ).astype(int)
            )

    points_layer.events.set_data.connect(on_move_point)

    def on_size_change(event):
        layer = event.source
        layer.size = layer.current_size

    points_layer.events.current_size.connect(on_size_change)

    view.add_layer(points_layer)
    qt_controls_container = view.window.qt_viewer.controls
    qt_controls_container.widgets[points_layer].layout().itemAt(3).widget().setText(
        "particle diameter (px)"
    )
    view.scale_bar.visible = True
    view.scale_bar.unit = "um"

    # Save annotations
    f = open(output_path, "w")

    def save_annotations():
        # delete previous annotations
        f.seek(0)
        f.truncate()

        # write new annotations
        s = points_layer.current_size
        f.write(",".join(["index", "axis-1", "axis-2", "axis-3", "size"]))
        f.write("\n")
        for i, pos in enumerate(points_layer.data):
            f.write(str(i) + ",")
            f.write(",".join(map(str, points_layer.data_to_world(pos))))
            f.write("," + str(s * spacing[1]))
            f.write("\n")
        f.tell()

    save_annotations()
    points_layer.events.data.connect(save_annotations)
    points_layer.events.current_size.connect(save_annotations)

    points_layer.mode = "add"
    napari.run()

    f.close()
