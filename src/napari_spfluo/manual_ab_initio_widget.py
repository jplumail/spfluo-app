import math
import sys

import napari
import napari.layers
import numpy as np
from napari.layers.shapes._shapes_constants import Mode
from napari.qt.threading import thread_worker
from napari.utils import notifications
from qtpy.QtWidgets import QDoubleSpinBox, QPushButton, QVBoxLayout, QWidget
from spfluo.utils.minimal_reconstruction import reconstruct
from spfluo.utils.volume import interpolate_to_size, resample


def get_scale(layer: napari.layers.Image):
    return np.asarray(layer.scale)


class ManualAbInitioWidget(QWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        top_particle_layer: "napari.layers.Image",
        side_particle_layer: "napari.layers.Image",
        psf_layer: "napari.layers.Image",
    ):
        super().__init__()

        self._viewer: napari.Viewer = viewer
        self._top_particle_layer = top_particle_layer
        self._side_particle_layer = side_particle_layer
        self._psf_layer = psf_layer
        self._lambda_widget = QDoubleSpinBox()
        self._lambda_widget.setRange(-6.0, 6.0)
        self._lambda_widget.setValue(0.0)

        self._reconstruction_layer = self._viewer.add_image(
            data=np.empty((1, 1)), name="_reconstruction", visible=False
        )

        self._shape_layer = self._viewer.add_shapes(name="_side_particle_axis")
        self._shape_layer.scale = top_particle_layer.scale[-2:]
        print(f"{id(self._viewer)=}", self._viewer.layers, file=sys.stderr)
        self._shape_layer.mode = Mode.ADD_LINE

        self._run_button_widget = QPushButton("Run")
        self._run_button_widget.clicked.connect(self._run_reconstruct)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._lambda_widget)
        self.layout().addWidget(self._run_button_widget)

    def _run_reconstruct(self):
        if len(self._shape_layer.data) == 0:
            notifications.show_error(
                "No data found in the _side_particle_axis layer.\n"
                "Please indicate the side particle axis before pressing the Run button."
            )
            return

        scale_side = get_scale(self._side_particle_layer)
        scale_top = get_scale(self._top_particle_layer)
        scale_psf = get_scale(self._psf_layer)
        min_scale = np.min(np.stack((scale_side, scale_top, scale_psf)))
        target_scale = np.asarray([min_scale, min_scale, min_scale])

        side_resampled = resample(
            self._side_particle_layer.data,
            scale_side / target_scale,
        )
        top_resampled = resample(
            self._top_particle_layer.data, scale_top / target_scale
        )
        psf_resampled = resample(
            self._psf_layer.data, scale_psf / target_scale
        )

        max_dim = max(max(top_resampled.shape), max(side_resampled.shape))
        size = (max_dim, max_dim, max_dim)

        top_resampled_cube = np.asarray(
            interpolate_to_size(top_resampled, size), dtype=np.float32
        )
        side_resampled_cube = np.asarray(
            interpolate_to_size(side_resampled, size), dtype=np.float32
        )
        psf_resampled_cube = np.asarray(
            interpolate_to_size(psf_resampled, size), dtype=np.float32
        )

        v0 = np.asarray([1.0, 0.0])
        v3 = np.asarray([0.0, 1.0])
        v1 = self._shape_layer.data[0][0] - self._shape_layer.data[0][1]
        theta = np.sign(np.dot(v3, v1)) * np.arccos(
            np.dot(v0, v1) / (np.linalg.norm(v1))
        )

        pose_side = np.asarray([theta * 180 / np.pi, 90.0, 0.0, 0.0, 0.0, 0.0])

        @thread_worker(progress=True)
        def _reconstruct():
            self._reconstruction_layer.data = reconstruct(
                top_resampled_cube,
                side_resampled_cube,
                pose_side,
                psf_resampled_cube,
                lambda_=math.pow(10, self._lambda_widget.value()),
            )
            self._reconstruction_layer.scale = target_scale
            self._reconstruction_layer.visible = True

        worker = _reconstruct()
        worker.start()
        return worker
