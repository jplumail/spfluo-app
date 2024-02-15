import os

import numpy as np
from numpy import fft
from scipy.ndimage import center_of_mass, fourier_shift


from ...utils.read_save_files import save
from ..common_image_processing_methods.others import crop_center
from ..common_image_processing_methods.registration import (
    center_connected_component,
    registration_exhaustive_search,
    shift_registration_exhaustive_search,
)


class Fourier_pixel_representation:
    def __init__(
        self, nb_dim, size, psf, init_vol=None, random_init=True, dtype=np.float32
    ):
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
        complex_type_promotion = {
            np.dtype("float32"): np.complex64,
            np.dtype("float64"): np.complex128,
        }
        self.volume_fourier = volume_fourier.astype(
            complex_type_promotion[np.dtype(dtype)]
        )
        self.nb_dim = nb_dim
        self.size = size
        self.psf = psf.astype(dtype)
        self.psf /= np.sum(self.psf)

    def gd_step(self, grad, lr, reg_coeff=0):
        self.volume_fourier -= lr * grad
        gradient_l2_reg, l2_reg = self.l2_regularization()
        self.volume_fourier -= lr * reg_coeff * gradient_l2_reg

    def l2_regularization(self):
        gradient_l2_reg = 2 * self.volume_fourier
        l2_reg = np.mean(np.abs(self.volume_fourier) ** 2)
        return gradient_l2_reg, l2_reg

    def get_image_from_fourier_representation(self):
        ifft = fft.ifftn(self.volume_fourier)
        image = np.abs(fft.fftshift(ifft)).real
        return image

    def save(self, output_dir, output_name):
        path = f"{output_dir}/{output_name}.tif"
        save(path, self.get_image_from_fourier_representation())

    def register_and_save(self, output_dir, output_name, ground_truth=None):
        im = self.get_image_from_fourier_representation()

        if ground_truth is not None:
            _, im = shift_registration_exhaustive_search(ground_truth, im)
            im = im.astype(ground_truth.dtype)
            _, im = registration_exhaustive_search(ground_truth, im)

        path = os.path.join(output_dir, output_name)
        save(path, im)
        return im

    def center(self):
        vol, shift = center_connected_component(
            self.get_image_from_fourier_representation()
        )
        com = np.asarray(center_of_mass(vol))
        c = (np.asarray(self.volume_fourier.shape) - 1) / 2
        shift += c - com
        self.volume_fourier = fourier_shift(self.volume_fourier, shift)
        return shift
