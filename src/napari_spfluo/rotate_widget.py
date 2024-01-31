import napari
import numpy as np
from magicgui.widgets import (
    ComboBox,
    Container,
    FloatSlider,
    create_widget,
)
from spfluo.utils.transform import get_transform_matrix
from spfluo.utils.volume import affine_transform


class RotateWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer: napari.Viewer = viewer

        self.rotated_particle_layer = None

        self._particle_layer_combo: ComboBox = create_widget(
            label="Particle", annotation="napari.layers.Image"
        )
        self._rotation_widgets = [
            FloatSlider(value=0, name="𝛗", min=0, max=360, tracking=True),
            FloatSlider(value=0, name="𝛉", min=0, max=180, tracking=True),
            FloatSlider(value=0, name="𝛙", min=0, max=360, tracking=True),
            FloatSlider(
                value=0,
                name="translation Z (px)",
                min=-10,
                max=10,
                tracking=True,
            ),
            FloatSlider(
                value=0,
                name="translation Y (px)",
                min=-10,
                max=10,
                tracking=True,
            ),
            FloatSlider(
                value=0,
                name="translation X (px)",
                min=-10,
                max=10,
                tracking=True,
            ),
        ]
        self._particle_layer_combo.changed.connect(self._on_layer_changed)
        for w in self._rotation_widgets:
            w.changed.connect(self._rotate)

        self.extend([self._particle_layer_combo])
        self.extend(self._rotation_widgets)

    def _on_layer_changed(self, layer):
        # change translation range
        for w, s in zip(self._rotation_widgets[3:], layer.data.shape):
            w.min = -s // 2
            w.max = s // 2

        if self.rotated_particle_layer is None:
            self.rotated_particle_layer = self._viewer.add_image(
                data=np.zeros(layer.data.shape), name="rotated particle"
            )

        # run rotation
        self._rotate()

    def _rotate(self):
        if self._particle_layer_combo.value is not None:
            if self.rotated_particle_layer is None:
                self.rotated_particle_layer = self._viewer.add_image(
                    data=np.zeros(self._particle_layer_combo.value.data.shape),
                    name="rotated particle",
                )

            rot_mat = get_transform_matrix(
                self.rotated_particle_layer.data.shape,
                np.asarray(
                    [
                        self._rotation_widgets[0].value,
                        self._rotation_widgets[1].value,
                        self._rotation_widgets[2].value,
                    ]
                ),
                np.asarray(
                    [
                        self._rotation_widgets[3].value,
                        self._rotation_widgets[4].value,
                        self._rotation_widgets[5].value,
                    ]
                ),
                degrees=True,
            )
            out = affine_transform(
                self._particle_layer_combo.value.data, rot_mat
            )
            self.rotated_particle_layer.data = out
