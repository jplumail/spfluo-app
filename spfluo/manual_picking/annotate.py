import csv
import os

import napari
import numpy as np
from napari.layers import Points
from napari.utils.events import Event

from spfluo.visualisation import add_orthoviewer_widget, init_qt


def annotate(im_path, output_path, size=10):
    init_qt()

    points_layer = Points(
        ndim=3,
        edge_color=[0, 0, 255, 255],
        face_color=[0, 0, 0, 0],
        out_of_slice_display=True,
        size=size,
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
                data.append((float(p[1]), float(p[2]), float(p[3])))
            data = np.array(data)
        points_layer.data = data

    view = napari.Viewer()
    view, dock_widget, cross = add_orthoviewer_widget(view)

    view.open(im_path, plugin="napari-aicsimageio")

    def on_move_point(event: Event):
        layer: Points = event.source
        viewers = [
            dock_widget.viewer,
            dock_widget.viewer_model1,
            dock_widget.viewer_model2,
        ]
        if len(layer.selected_data) > 0:
            idx_point = list(layer.selected_data)[0]
            pos_point = tuple(layer.data[idx_point])

            # update viewers
            viewers_not_under_mouse = [
                viewer for viewer in viewers if not viewer.mouse_over_canvas
            ]
            if (
                len(viewers_not_under_mouse) == 3
            ):  # if mouse is not over any viewer, the point size is being adjusted
                return
            for viewer in viewers_not_under_mouse:
                pos_reordered = tuple(np.array(pos_point)[list(viewer.dims.order)])
                viewer.camera.center = pos_reordered

            dock_widget.viewer.dims.current_step = tuple(
                np.round(
                    [
                        max(min_, min(p, max_)) / step
                        for p, (min_, max_, step) in zip(
                            pos_point, dock_widget.viewer.dims.range
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
            f.write(",".join(map(str, pos)))
            f.write("," + str(s))
            f.write("\n")
        f.tell()

    save_annotations()
    points_layer.events.set_data.connect(save_annotations)

    points_layer.mode = "add"
    napari.run()

    f.close()
