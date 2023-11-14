from napari import Viewer, run

from napari_spfluo.ab_initio_widget import AbInitioWidget

viewer = Viewer()
viewer.open_sample("napari-spfluo", "anisotropic")
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-spfluo", "Container Ab initio reconstruction"
)

# Optional steps to setup your plugin to a state of failure
# E.g. plugin_widget.parameter_name.value = "some value"
# E.g. plugin_widget.button.click()

plugin_widget: AbInitioWidget
plugin_widget._psf_layer_combo.value = viewer.layers[1]
run()
