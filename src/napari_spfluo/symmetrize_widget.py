import napari
import numpy as np
from magicgui.widgets import (
    ComboBox,
    Container,
    FloatSlider,
    PushButton,
    create_widget,
)
from skimage.util import img_as_float
from spfluo.utils.symmetrize_particle import symmetrize


class SymmetrizeWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer: napari.Viewer = viewer
        self._center_layer = self._viewer.add_points(
            ndim=2, name="_center", face_color="transparent"
        )

        self.symmetric_particle = None

        self._particle_layer_combo: ComboBox = create_widget(
            label="Particle", annotation="napari.layers.Image"
        )
        self._psf_layer_combo: ComboBox = create_widget(
            label="PSF", annotation="napari.layers.Image"
        )
        self._lambda_widget = FloatSlider(
            value=1,
            name="regularization (log10)",
            min=-10,
            max=10,
            tracking=True,
        )

        self._run_button = PushButton(text="Run")
        self._run_button.changed.connect(self._run_symmetrize)
        self._lambda_widget.changed.connect(self._run_symmetrize)

        self._particle_layer_combo.changed.connect(self._on_layer_changed)

        self.extend(
            [
                self._particle_layer_combo,
                self._psf_layer_combo,
                self._lambda_widget,
                self._run_button,
            ]
        )

    def _on_layer_changed(self, val):
        self._center_layer.scale = val.scale[1:]

    def _run_symmetrize(self):
        if len(self._center_layer.data) > 0:
            center = self._center_layer.world_to_data(
                np.asarray(self._center_layer.data[0])
                - np.asarray(self._particle_layer_combo.value.data.shape[1:])
                / 2
            )
            res = symmetrize(
                img_as_float(self._particle_layer_combo.value.data),
                (center[0], center[1]),
                9,
                img_as_float(self._psf_layer_combo.value.data),
                np.asarray(10**self._lambda_widget.value),
            )
            if self.symmetric_particle is None:
                self.symmetric_particle = self._viewer.add_image(
                    res, name="symmetric particle"
                )
            else:
                self.symmetric_particle.data = res

            self.symmetric_particle.reset_contrast_limits()
