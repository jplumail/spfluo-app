import napari
import numpy as np
from napari_bbox import BoundingBoxLayer

from spfluo.visualisation.multiple_viewer_widget import add_orthoviewer_widget


def create_multiviewer_with_data(viewer):
    _, widget, _ = add_orthoviewer_widget(viewer)
    viewer.window.add_dock_widget(widget)
    image_layer = viewer.add_image(np.random.randn(10, 100, 100))

    return widget, image_layer


def add_bbox_layer(viewer):
    bbox_layer = BoundingBoxLayer(ndim=3)
    viewer.add_layer(bbox_layer)
    bbox_layer.add(
        [
            [1, 1, 1],
            [10, 1, 1],
            [1, 10, 1],
            [1, 1, 10],
            [10, 10, 1],
            [10, 1, 10],
            [1, 10, 10],
            [10, 10, 10],
        ]
    )
    return bbox_layer


def test_data(make_napari_viewer_proxy):
    viewer: napari.Viewer = make_napari_viewer_proxy(show=True)
    widget, image_layer = create_multiviewer_with_data(viewer)
    add_bbox_layer(viewer)

    assert (
        len(widget.viewer.layers)
        == len(widget.viewer_model1.layers)
        == len(widget.viewer_model2.layers)
    )
    np.testing.assert_allclose(
        widget.viewer.layers[1].data, widget.viewer_model1.layers[1].data
    )
    np.testing.assert_allclose(
        widget.viewer.layers[1].data, widget.viewer_model2.layers[1].data
    )


def test_data_after_move(make_napari_viewer_proxy):
    viewer: napari.Viewer = make_napari_viewer_proxy(show=True)
    widget, image_layer = create_multiviewer_with_data(viewer)
    bbox_layer = add_bbox_layer(viewer)

    # move bbox in main viewer
    bbox_layer.data = [bbox_layer.data[0] + np.asarray([0.0, 5.0, 5.0])]

    assert (
        len(widget.viewer.layers)
        == len(widget.viewer_model1.layers)
        == len(widget.viewer_model2.layers)
    )
    np.testing.assert_allclose(
        widget.viewer.layers[1].data, widget.viewer_model1.layers[1].data
    )
    np.testing.assert_allclose(
        widget.viewer.layers[1].data, widget.viewer_model2.layers[1].data
    )


def test_remove(make_napari_viewer_proxy):
    viewer: napari.Viewer = make_napari_viewer_proxy(show=False)
    widget, image_layer = create_multiviewer_with_data(viewer)
    bbox_layer = add_bbox_layer(viewer)

    # select+remove
    bbox_layer.selected_data = {0}
    bbox_layer.remove_selected()

    assert (
        len(widget.viewer.layers)
        == len(widget.viewer_model1.layers)
        == len(widget.viewer_model2.layers)
    )
    np.testing.assert_allclose(
        widget.viewer.layers[1].data, widget.viewer_model1.layers[1].data
    )
    np.testing.assert_allclose(
        widget.viewer.layers[1].data, widget.viewer_model2.layers[1].data
    )


def test_remove2(make_napari_viewer_proxy):
    viewer: napari.Viewer = make_napari_viewer_proxy(show=True)
    widget, image_layer = create_multiviewer_with_data(viewer)
    bbox_layer = add_bbox_layer(viewer)
    bbox_layer.add(
        [
            [12, 12, 12],
            [20, 12, 12],
            [12, 20, 12],
            [12, 12, 20],
            [20, 20, 12],
            [20, 12, 20],
            [12, 20, 20],
            [20, 20, 20],
        ]
    )

    # select+remove
    bbox_layer.selected_data = {0}
    bbox_layer.remove_selected()

    assert (
        len(widget.viewer.layers)
        == len(widget.viewer_model1.layers)
        == len(widget.viewer_model2.layers)
    )
    np.testing.assert_allclose(
        widget.viewer.layers[1].data, widget.viewer_model1.layers[1].data
    )
    np.testing.assert_allclose(
        widget.viewer.layers[1].data, widget.viewer_model2.layers[1].data
    )
