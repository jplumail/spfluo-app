from importlib import import_module, reload

import napari
import napari.qt.threading
import numpy as np
import pytest

from napari_spfluo.manual_ab_initio_widget import ManualAbInitioWidget
from napari_spfluo.symmetrize_widget import SymmetrizeWidget


def test_symmetrize_widget(make_napari_viewer):
    viewer: napari.Viewer = make_napari_viewer()
    viewer.open_sample("napari-spfluo", "anisotropic")
    viewer.add_image(viewer.layers["volumes"].data[6], name="particle")

    widget = SymmetrizeWidget(viewer)
    widget._particle_layer_combo.value = viewer.layers["particle"]
    widget._psf_layer_combo.value = viewer.layers["psf"]

    widget._center_layer.add((25, 25))

    widget._run_symmetrize()
    assert widget.symmetric_particle in viewer.layers


@pytest.mark.xfail
def test_manual_ab_initio_widget(make_napari_viewer):
    # replace the @thread_worker decorator with a mock function
    def mock(function=None, **kwargs):
        def _inner(func):
            def worker_function(*args, **kwargs):
                class A:
                    def start(self):
                        func()

                return A()

            return worker_function

        return _inner if function is None else _inner(function)

    napari.qt.threading.thread_worker = mock
    # reload the module for the replaced decorator to take effect
    module_name = ManualAbInitioWidget.__module__
    module = import_module(module_name)
    reload(module)
    assert module.thread_worker == mock

    viewer: napari.Viewer = make_napari_viewer()
    viewer.open_sample("napari-spfluo", "anisotropic")

    viewer.layers["volumes"].visible = False

    side_layer = viewer.add_image(
        viewer.layers["volumes"].data[3], name="side"
    )
    top_layer = viewer.add_image(
        viewer.layers["volumes"].data[6], name="top", visible=False
    )
    psf_layer = viewer.layers["psf"]
    psf_layer.visible = False

    widget = ManualAbInitioWidget(viewer, top_layer, side_layer, psf_layer)

    widget._shape_layer.data = [
        np.asarray([[32.72102977, 40.65432298], [14.96360665, 4.64621497]])
    ]

    worker = widget._run_reconstruct()
    worker.start()
    assert max(widget._reconstruction_layer.data.shape) > 1
    assert widget._reconstruction_layer.visible
