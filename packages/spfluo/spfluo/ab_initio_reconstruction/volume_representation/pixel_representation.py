from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from spfluo.utils.array import array_namespace, to_numpy
from spfluo.utils.volume import center_of_mass, fourier_shift

from ...utils.read_save_files import save_image
from ..common_image_processing_methods.others import crop_center
from ..common_image_processing_methods.registration import (
    center_connected_component,
    registration_exhaustive_search,
    shift_registration_exhaustive_search,
)

if TYPE_CHECKING:
    from spfluo.utils.array import Array


class Fourier_pixel_representation:
    def __init__(
        self,
        nb_dim: int,
        size: tuple[int, ...],
        psf: "Array",
        init_vol: "Array" | None = None,
        random_init: bool = True,
        dtype: str = "float32",
    ):
        xp = array_namespace(psf)
        self.xp = xp
        self.dtype = getattr(xp, dtype)
        complex_type_promotion = {
            "float32": "complex64",
            "float64": "complex128",
        }
        self.complex_dtype = getattr(xp, complex_type_promotion[dtype])
        if init_vol is None:
            if random_init:
                volume_fourier = np.random.randn(*psf.shape) + 1j * np.random.randn(
                    *psf.shape
                )
            else:
                raise NotImplementedError
        else:
            volume_fourier = np.fft.fftn(
                np.fft.ifftshift(crop_center(np.asarray(init_vol), (size, size, size)))
            )
        self.volume_fourier = xp.asarray(volume_fourier, dtype=self.complex_dtype)
        self.nb_dim = nb_dim
        self.size = size
        self.psf = xp.asarray(psf, dtype=self.dtype)
        self.psf /= xp.sum(self.psf)

    def gd_step(self, grad, lr, reg_coeff=0):
        self.volume_fourier -= lr * grad
        gradient_l2_reg, l2_reg = self.l2_regularization()
        self.volume_fourier -= lr * reg_coeff * gradient_l2_reg

    def l2_regularization(self):
        gradient_l2_reg = 2 * self.volume_fourier
        l2_reg = self.xp.mean(self.xp.abs(self.volume_fourier) ** 2)
        return gradient_l2_reg, l2_reg

    def get_image_from_fourier_representation(self):
        ifft = self.xp.fft.ifftn(self.volume_fourier)
        image = self.xp.abs(self.xp.fft.fftshift(ifft)).real
        return image

    def save(self, output_dir, output_name):
        path = f"{output_dir}/{output_name}.ome.tiff"
        save_image(path, self.get_image_from_fourier_representation(), order="ZYX")

    def register_and_save(self, output_dir, output_name, ground_truth=None):
        im = self.get_image_from_fourier_representation()

        if ground_truth is not None:
            _, im = shift_registration_exhaustive_search(ground_truth, im)
            im = im.astype(ground_truth.dtype)
            _, im = registration_exhaustive_search(ground_truth, im)

        path = os.path.join(output_dir, output_name)
        save_image(path, im, order="ZYX")
        return im

    def center(self):
        vol, shift = center_connected_component(
            to_numpy(self.get_image_from_fourier_representation())
        )
        shift = self.xp.asarray(shift)
        com = self.xp.asarray(center_of_mass(self.xp.asarray(vol)))
        c = (self.xp.asarray(self.volume_fourier.shape) - 1) / 2
        shift += c - com
        self.volume_fourier = self.xp.astype(
            fourier_shift(self.volume_fourier, shift), self.complex_dtype
        )
        return shift
