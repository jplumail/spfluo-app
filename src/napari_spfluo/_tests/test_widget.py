import napari

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
