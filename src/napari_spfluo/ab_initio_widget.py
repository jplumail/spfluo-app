from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import (
    ComboBox,
    Container,
    PushButton,
    create_widget,
)
from napari.qt.threading import thread_worker
from skimage.util import img_as_float
from spfluo.ab_initio_reconstruction.api import AbInitioReconstruction
from spfluo.utils.volume import interpolate_to_size

if TYPE_CHECKING:
    import napari
    from napari.viewer import Viewer


class AbInitioWidget(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self._viewer: Viewer = viewer
        self._particles_layer_combo: ComboBox = create_widget(
            label="Particles", annotation="napari.layers.Image"
        )
        self._psf_layer_combo: ComboBox = create_widget(
            label="PSF", annotation="napari.layers.Image"
        )

        self._run_button = PushButton(text="Run")
        self._run_button.changed.connect(self._run_ab_initio)

        # self._progress_bar = ProgressBar(min=0, max=5)

        self.extend(
            [
                self._particles_layer_combo,
                self._psf_layer_combo,
                self._run_button,
                # self._progress_bar
            ]
        )

    def _run_ab_initio(self):
        particles_array = img_as_float(self._particles_layer_combo.value.data)
        psf_array = img_as_float(self._psf_layer_combo.value.data)
        assert particles_array.shape == (10, 50, 50, 50)
        psf_array = interpolate_to_size(psf_array, particles_array.shape[1:])
        assert psf_array.shape == (50, 50, 50)
        reconstruction_layer = self._viewer.add_image(
            np.zeros_like(particles_array[0]), name="reconstruction"
        )

        def callback(arr: np.ndarray, iteration: int):
            reconstruction_layer.data = arr
            # self._progress_bar.increment(float(iteration))

        ab_initio = AbInitioReconstruction(N_iter_max=5, callback=callback)

        @thread_worker
        def fitting() -> AbInitioReconstruction:
            ab_initio.fit(particles_array, psf=psf_array, gpu="pytorch")
            return ab_initio._volume

        worker = fitting()
        worker.start()

        return ab_initio._volume


def run_ab_initio(
    particles: napari.types.ImageData,
    psf: napari.types.ImageData,
    M_axes: int = 360**2,
    M_rot: int = 360,
    dec_prop: float = 1.2,
    init_unif_prop: tuple[int, int] = (1, 1),
    coeff_kernel_axes: float = 50.0,
    coeff_kernel_rot: float = 5.0,
    eps: float = 0,
    lr: float = 0.1,
    N_axes: int = 25,
    N_rot: int = 20,
    prop_min: float = 0,
    interp_order: int = 3,
    N_iter_max: int = 20,
    gaussian_kernel: bool = True,
    convention: str = "XZX",
    dtype: np.dtype = np.float32,
    reg_coeff: float = 0,
    beta_sampling: float = 0,
    batch_size: int = 1,
    beta_grad: float = 0,
    random_sampling: bool = False,
) -> napari.types.ImageData:
    ab_initio = AbInitioReconstruction(
        M_axes=M_axes,
        M_rot=M_rot,
        dec_prop=dec_prop,
        init_unif_prop=init_unif_prop,
        coeff_kernel_axes=coeff_kernel_axes,
        coeff_kernel_rot=coeff_kernel_rot,
        eps=eps,
        lr=lr,
        N_axes=N_axes,
        N_rot=N_rot,
        prop_min=prop_min,
        interp_order=interp_order,
        N_iter_max=N_iter_max,
        gaussian_kernel=gaussian_kernel,
        convention=convention,
        dtype=dtype,
        reg_coeff=reg_coeff,
        beta_sampling=beta_sampling,
        batch_size=batch_size,
        beta_grad=beta_grad,
        random_sampling=random_sampling,
    )
    particles_array = img_as_float(particles)
    psf_array = img_as_float(psf)
    assert particles_array.shape == (10, 50, 50, 50)
    print(psf_array.shape, particles_array.shape)
    psf_array = interpolate_to_size(psf_array, particles_array.shape[1:])
    assert psf_array.shape == (50, 50, 50)

    @thread_worker
    def fitting() -> AbInitioReconstruction:
        ab_initio.fit(particles_array, psf=psf_array)
        return ab_initio

    # worker = fitting()
    # worker.returned.connect(viewer.add_image)

    return ab_initio._volume
