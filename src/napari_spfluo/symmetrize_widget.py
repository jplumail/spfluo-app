from magicgui.widgets import PushButton, create_widget, Container, ComboBox
import napari
import torch
import numpy as np

from spfluo.utils.symmetrize_particle import symmetrize
from skimage.util import img_as_float


class SymmetrizeWidget(Container):
    def __init__(self, viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self._viewer: napari.Viewer = viewer
        self._center_layer = self._viewer.add_points(ndim=2, name="_center", face_color="transparent")

        self.symmetric_particle = None

        self._particle_layer_combo: ComboBox = create_widget(
            label="Particle", annotation="napari.layers.Image"
        )
        self._psf_layer_combo: ComboBox = create_widget(
            label="PSF", annotation="napari.layers.Image"
        )
        self._lambda_widget = create_widget(value=1e-2)

        self._run_button = PushButton(text="Run")
        self._run_button.changed.connect(self._run_symmetrize)

        self.extend([
            self._particle_layer_combo,
            self._psf_layer_combo,
            self._lambda_widget,
            self._run_button,
        ])
    
    def _run_symmetrize(self):
        center = np.asarray(self._center_layer.data[0]) - np.asarray(self._particle_layer_combo.value.data.shape[1:])/2
        print(center)
        res = symmetrize(
            torch.as_tensor(img_as_float(self._particle_layer_combo.value.data)),
            (center[0], center[1]),
            9,
            torch.as_tensor(img_as_float(self._psf_layer_combo.value.data)),
            torch.as_tensor(self._lambda_widget.value)
        ).numpy()
        if self.symmetric_particle is None:
            self.symmetric_particle = self._viewer.add_image(res, name="symmetric particle")
        else:
            self.symmetric_particle.data = res