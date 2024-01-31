import napari
import numpy as np
from magicgui.widgets import (
    ComboBox,
    Container,
    FloatSlider,
    PushButton,
    create_widget,
)
from napari.layers import Image
from spfluo.utils.separate import separate_centrioles


class SeparateWidget(Container):
    """Centioles are often glued together.
    It's easy to pick them 2 by 2.
    Then we need to separate them, that's what this widget does.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer: napari.Viewer = viewer

        self.output1_particle_layer = None
        self.output2_particle_layer = None

        self._particles_layer_combo: ComboBox = create_widget(
            label="2 particles", annotation="napari.layers.Image"
        )

        self._threshold_widget = FloatSlider(
            value=0.5, name="threshold value", min=0, max=1, tracking=False
        )
        self._threshold_widget.changed.connect(self._separate)
        self._particles_layer_combo.changed.connect(self._on_layer_changed)
        self._run_button = PushButton(text="Run")
        self._run_button.changed.connect(self._separate)

        self.extend(
            [
                self._particles_layer_combo,
                self._threshold_widget,
                self._run_button,
            ]
        )

    def _init_output_layers(self, layer: Image):
        self.output1_particle_layer = self._viewer.add_image(
            data=np.zeros(layer.data.shape),
            name="rotated particle",
            scale=layer.scale,
        )
        self.output2_particle_layer = self._viewer.add_image(
            data=np.zeros(layer.data.shape),
            name="rotated particle",
            scale=layer.scale,
        )

    def _on_layer_changed(self, layer: Image):
        if self.output1_particle_layer is None:
            self._init_output_layers(layer)

        # run rotation
        self._separate()

    def _separate(self):
        if self._particles_layer_combo.value is not None:
            if self.output1_particle_layer is None:
                self._init_output_layers(self._particles_layer_combo.value)

            im = self._particles_layer_combo.value.data
            im1, im2 = separate_centrioles(
                im, threshold_percentage=self._threshold_widget.value
            )

            self.output1_particle_layer.data = im1
            self.output2_particle_layer.data = im2
            self.output1_particle_layer.reset_contrast_limits()
            self.output2_particle_layer.reset_contrast_limits()
