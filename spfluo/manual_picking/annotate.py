import atexit
import csv
import itertools
import os
from typing import TYPE_CHECKING, Tuple

import napari
import numpy as np
from napari.utils.events import Event
from napari_bbox import BoundingBoxLayer

from spfluo.visualisation.multiple_viewer_widget import add_orthoviewer_widget, init_qt

if TYPE_CHECKING:
    from napari.components import ViewerModel


def annotate(
    im_path: str,
    output_path: str,
    spacing: Tuple[float, float, float] = (1, 1, 1),
    save: bool = True,
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
    # correct spacing, see https://github.com/napari/napari/issues/6627
    spacing_normalized = np.asarray(spacing) / np.min(spacing)
    edge_width = max(
        (
            0.4  # 0.4um: length of a centriole
            / 10  # 1/10th of the length of a centriole
            / spacing[1]  # in pixel space
        ),
        2,
    )

    init_qt()

    bbox_layer = BoundingBoxLayer(
        ndim=3,
        edge_color=[0, 0, 255, 255],
        face_color=[0, 0, 0, 0],
        scale=spacing_normalized,
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
                p1 = bbox_layer.world_to_data(
                    np.asarray(
                        (float(p[0 * 3 + 1]), float(p[0 * 3 + 2]), float(p[0 * 3 + 3]))
                    )
                    / np.min(spacing)
                )
                p2 = bbox_layer.world_to_data(
                    np.asarray(
                        (float(p[1 * 3 + 1]), float(p[1 * 3 + 2]), float(p[1 * 3 + 3]))
                    )
                    / np.min(spacing)
                )
                data.append(
                    list(
                        itertools.product(
                            (p1[0], p2[0]), (p1[1], p2[1]), (p1[2], p2[2])
                        )
                    )
                )
            data = np.asarray(data)
        bbox_layer.data = data

    view = napari.Viewer()
    view, dock_widget, cross = add_orthoviewer_widget(view)

    view.open(im_path, layer_type="image", scale=spacing_normalized)

    viewers: list[ViewerModel] = [
        dock_widget.viewer,
        dock_widget.viewer_model1,
        dock_widget.viewer_model2,
    ]

    def update_viewer(layer: BoundingBoxLayer, event: Event):
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if len(layer.selected_data) > 0:
            idx_point = list(layer.selected_data)[0]
            if dragged:
                layer.data = layer.data  # seems stupid but very important
            try:
                bbox = tuple(layer.data[idx_point])
            except IndexError:
                return
            bbox_center = np.array(bbox).mean(axis=0)
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
                    np.array(bbox_layer.data_to_world(bbox_center))[
                        list(viewer.dims.order)
                    ]
                )
                viewer.camera.center = pos_reordered

            dock_widget.viewer.dims.current_step = tuple(
                np.round(
                    [
                        max(min_, min(p, max_)) / step
                        for p, (min_, max_, step) in zip(
                            bbox_layer.data_to_world(bbox_center),
                            dock_widget.viewer.dims.range,
                        )
                    ]
                ).astype(int)
            )

    view.add_layer(bbox_layer)
    for v in viewers:
        for layer in v.layers:
            if isinstance(layer, BoundingBoxLayer):
                layer.mouse_drag_callbacks.append(update_viewer)
                layer.edge_width = edge_width
    view.scale_bar.visible = True
    view.scale_bar.unit = "um"

    # Save annotations
    if save:
        f = open(output_path, "w")

        def save_annotations():
            # delete previous annotations
            f.seek(0)
            f.truncate()

            # write new annotations
            f.write(
                ",".join(
                    [
                        "index",
                        "axis-1",
                        "axis-2",
                        "axis-3",
                        "axis-1",
                        "axis-2",
                        "axis-3",
                    ]
                )
            )
            f.write("\n")
            for i, bbox in enumerate(bbox_layer.data):
                f.write(str(i) + ",")
                f.write(
                    ",".join(
                        map(
                            str,
                            np.asarray(bbox_layer.data_to_world(np.min(bbox, axis=0)))
                            * np.min(spacing),
                        )
                    )
                )
                f.write(",")
                f.write(
                    ",".join(
                        map(
                            str,
                            np.asarray(bbox_layer.data_to_world(np.max(bbox, axis=0)))
                            * np.min(spacing),
                        )
                    )
                )
                f.write("\n")
            f.tell()

        save_annotations()
        bbox_layer.events.data.connect(save_annotations)

        bbox_layer.mode = "ADD_BOUNDING_BOX"
        atexit.register(lambda: f.close())

    napari.run()
