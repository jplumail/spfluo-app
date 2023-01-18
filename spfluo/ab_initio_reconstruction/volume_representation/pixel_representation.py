from spfluo.utils import dftregistrationND as phase_cross_correlation_gpu_pytorch, fourier_shift as fourier_shift_gpu_pytorch
from ..common_image_processing_methods.rotation_translation import rotation, translation, rotation_gpu, rotation_gpu_pytorch
from ..common_image_processing_methods.registration import registration_exhaustive_search, shift_registration_exhaustive_search
from ..common_image_processing_methods.others import normalize, crop_center
from ..common_image_processing_methods.registration import translate_to_have_one_connected_component
#from ..common_image_processing_methods.barycenter import center_barycenter
from ..manage_files.read_save_files import save, read_image

import os
import numpy as np
import cupy as cp
from numpy import fft
import torch
from skimage.registration import phase_cross_correlation
from cucim.skimage.registration import phase_cross_correlation as phase_cross_correlation_gpu
from scipy.ndimage.fourier import fourier_shift
from cupyx.scipy.ndimage import fourier_shift as fourier_shift_gpu
from time import time


class Fourier_pixel_representation:
    def __init__(self, nb_dim, size, psf, init_vol=None, random_init=True, dtype=np.float32):
        if init_vol is None:
            if random_init:
                volume_fourier = np.random.randn(*psf.shape) + 1j * np.random.randn(*psf.shape)
            else:
                raise NotImplementedError
        else:
            volume_fourier = np.fft.fftn(np.fft.ifftshift(crop_center(np.asarray(init_vol), (size, size, size))))
        self.volume_fourier = volume_fourier.astype(complex)
        self.volume_fourier_gpu = cp.array(self.volume_fourier)
        self.volume_fourier_gpu_torch = torch.as_tensor(self.volume_fourier)
        self.nb_dim = nb_dim
        self.size = size
        self.psf = psf.astype(dtype)
        self.psf_gpu = cp.array(self.psf)
        self.psf_gpu_torch = torch.as_tensor(self.psf)
        self.psf_fft = fft.fftn(self.psf)

    def get_energy(self, rot_mat, trans_vec, view, recorded_shifts, view_idx, save_shift=False, known_trans=True, interp_order=3):
        psf_rotated_fft = fft.fftn(rotation(self.psf, rot_mat.T)[0])
        if known_trans:

            view_rotated_fft = fft.fftn(rotation(view, rot_mat.T, trans_vec=-rot_mat.T@trans_vec, order=interp_order)[0])
        else:
            view_rotated_fft = fft.fftn(rotation(view, rot_mat.T, order=interp_order)[0])
            shift, _, _ = phase_cross_correlation(psf_rotated_fft*self.volume_fourier, view_rotated_fft, space='fourier', upsample_factor=10, normalization=None)
            view_rotated_fft = fourier_shift(view_rotated_fft, shift)
            if save_shift:
                recorded_shifts[view_idx].append(-rot_mat@shift)
        energy = np.linalg.norm(psf_rotated_fft*self.volume_fourier - view_rotated_fft)**2/(self.size**self.nb_dim)
        variables_used_to_compute_gradient = [psf_rotated_fft, view_rotated_fft]
        return energy, variables_used_to_compute_gradient
    
    def get_energy_gpu(self, rot_mat, trans_vec, view, recorded_shifts, view_idx, save_shift=False, known_trans=True, interp_order=3):
        psf_rotated = rotation_gpu(self.psf_gpu, rot_mat.T)[0]
        psf_rotated_fft = cp.fft.fftn(psf_rotated)
        if known_trans:

            view_rotated_fft = cp.fft.fftn(rotation_gpu(view, rot_mat.T, trans_vec=-rot_mat.T@trans_vec, order=interp_order)[0])
        else:
            view_rotated_fft = cp.fft.fftn(rotation_gpu(view, rot_mat.T, order=interp_order)[0])
            shift, _, _ = phase_cross_correlation_gpu(psf_rotated_fft*self.volume_fourier_gpu, view_rotated_fft, space='fourier', upsample_factor=10, normalization=None)

            view_rotated_fft = fourier_shift_gpu(view_rotated_fft, shift)
            if save_shift:
                recorded_shifts[view_idx].append(-rot_mat@shift)
        energy = cp.linalg.norm(psf_rotated_fft*self.volume_fourier_gpu - view_rotated_fft)**2/(self.size**self.nb_dim)
        variables_used_to_compute_gradient = (psf_rotated_fft, view_rotated_fft)
        return energy, variables_used_to_compute_gradient
    
    def get_energy_gpu_pytorch(self, rot_mats, trans_vec, view, recorded_shifts, view_idx, save_shift=False, known_trans=True, interp_order=1):
        N = rot_mats.size(0)
        psfs_rotated = rotation_gpu_pytorch(self.psf_gpu_torch.to(rot_mats.device).repeat(N,1,1,1,1), rot_mats.permute((0,2,1)), order=interp_order)[0][:,0]
        psfs_rotated_fft = torch.fft.fftn(psfs_rotated, dim=(1,2,3))
        if known_trans:

            view_rotated_fft = torch.fft.fftn(rotation_gpu_pytorch(view.repeat(N,1,1,1,1), rot_mats.permute((0,2,1)), trans_vec=-rot_mats.permute((0,2,1))@trans_vec, order=interp_order)[0][:,0], dim=(1,2,3))
        else:
            view_rotated_fft = torch.fft.fftn(rotation_gpu_pytorch(view.repeat(N,1,1,1,1), rot_mats.permute((0,2,1)), order=interp_order)[0][:,0], dim=(1,2,3))
            _, shift = phase_cross_correlation_gpu_pytorch(psfs_rotated_fft*self.volume_fourier_gpu_torch.to(rot_mats.device), view_rotated_fft, nb_spatial_dims=3, upsample_factor=10, normalization=None)
            shift = torch.stack(shift, dim=-1)

            view_rotated_fft = fourier_shift_gpu_pytorch(view_rotated_fft, shift, nb_spatial_dims=3)
            if save_shift:
                recorded_shifts[view_idx].append(-rot_mats.permute((0,2,1))@shift)
        energy = torch.linalg.norm((psfs_rotated_fft*self.volume_fourier_gpu_torch.to(rot_mats.device) - view_rotated_fft).view(psfs_rotated_fft.size(0),-1), dim=1)**2/(self.size**self.nb_dim)
        variables_used_to_compute_gradient = [psfs_rotated_fft, view_rotated_fft]
        return energy, variables_used_to_compute_gradient

    def get_energy_2(self, rot_mat, trans_vec, view_fft, known_trans=True):
        rotated_volume = rotation(self.volume_fourier, rot_mat)[0]
        convolved_rotated_volume = self.psf_fft * rotated_volume
        if not known_trans:
            shift, _, _ = phase_cross_correlation(convolved_rotated_volume, view_fft)
        else:
            shift = trans_vec
        translated_convolved_rotated_volume = fourier_shift(convolved_rotated_volume, shift)
        energy = np.linalg.norm(translated_convolved_rotated_volume-view_fft)**2/(self.size*self.nb_dim)
        return energy


    def compute_grad(self, rot_mat, trans_vec, views, known_trans, view_idx, interp_order=3):
        view = views[view_idx]
        energy, variables_used_to_compute_gradient = \
            self.get_energy(rot_mat, trans_vec, view, None, view_idx,
                            known_trans=known_trans, interp_order=interp_order)
        psf_rotated_fft, view_rotated_fft = variables_used_to_compute_gradient
        grad = psf_rotated_fft * (psf_rotated_fft*self.volume_fourier-view_rotated_fft)
        return grad, energy
    
    def gd_step(self, grad, lr, reg_coeff=0):
        self.volume_fourier -= lr*grad
        gradient_l2_reg, l2_reg  = self.l2_regularization()
        self.volume_fourier -= lr*reg_coeff*gradient_l2_reg
        self.volume_fourier_gpu = cp.array(self.volume_fourier)
        self.volume_fourier_gpu_torch = torch.as_tensor(self.volume_fourier)

    def one_gd_step(self, rot_mat, trans_vec, views, lr, known_trans, view_idx, recorded_shifts, reg_coeff=0, ground_truth=None, interp_order=3):
        view = views[view_idx]
        energy, variables_used_to_compute_gradient = \
            self.get_energy(rot_mat, trans_vec, view, recorded_shifts, view_idx, save_shift=True,
                            known_trans=known_trans, interp_order=interp_order)
        psf_rotated_fft, view_rotated_fft = variables_used_to_compute_gradient
        grad = psf_rotated_fft * (psf_rotated_fft*self.volume_fourier-view_rotated_fft)
        self.volume_fourier -= lr*grad
        gradient_l2_reg, l2_reg  = self.l2_regularization()
        self.volume_fourier -= lr*reg_coeff*gradient_l2_reg
        self.volume_fourier_gpu = cp.array(self.volume_fourier)
        self.volume_fourier_gpu_torch = torch.as_tensor(self.volume_fourier)
        if not known_trans and ground_truth is not None:
            # _, self.volume_fourier = shift_registration_exhaustive_search(np.fft.fftn(ground_truth), self.volume_fourier, fourier_space=True)
            #self.volume_fourier = center_barycenter(self.volume_fourier)
            pass
        energy += reg_coeff*l2_reg
        return energy

    def squared_variations_regularization(self):
        v1 = self.volume_fourier[1:, :, :]
        v2 = self.volume_fourier[:, 1:, :]
        v3 = self.volume_fourier[:, :, 1:]
        v1 = np.pad(v1, ((0,1), (0,0), (0,0)))
        v2 = np.pad(v2, ((0,0), (0,1), (0,0)))
        v3 = np.pad(v3, ((0, 0), (0, 0), (0,1)))
        TV = np.mean((self.volume_fourier-v1)**2) + np.mean((self.volume_fourier-v2)**2) + np.mean((self.volume_fourier-v3)**2)
        gradient_TV = 2*self.volume_fourier*(3*self.volume_fourier - v1 - v2 - v3)/self.size**3
        return TV, gradient_TV

    def l2_regularization(self):
        gradient_l2_reg = 2*self.volume_fourier
        l2_reg = np.mean(np.abs(self.volume_fourier)**2)
        return gradient_l2_reg, l2_reg

    def get_image_from_fourier_representation(self):
        ifft = fft.ifftn(self.volume_fourier)
        image = np.abs(fft.fftshift(ifft)).real
        return image

    def save(self, output_dir, output_name):
        path = f'{output_dir}/{output_name}.tif'
        save(path, self.get_image_from_fourier_representation())

    def register_and_save(self, output_dir, output_name, ground_truth=None, translate=False):
        path = os.path.join(output_dir, output_name)
        im = self.get_image_from_fourier_representation()
        if translate:
            im = translate_to_have_one_connected_component(im)

        if ground_truth is not None:
            _, im = shift_registration_exhaustive_search(ground_truth, im)
            im = im.astype(ground_truth.dtype)

        save(path, im)
        if ground_truth is not None:
            registration_exhaustive_search(ground_truth, im, output_dir, output_name)




if __name__ == '__main__':

    from scipy import ndimage, misc
    import matplotlib.pyplot as plt
    import numpy.fft
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.gray()  # show the filtered result in grayscale
    ascent = misc.ascent()
    input_ = numpy.fft.fft2(ascent)
    result = ndimage.fourier_shift(input_, shift=200)
    result = numpy.fft.ifft2(result)
    result = np.abs(fft.fftshift(result))

    ax1.imshow(ascent)
    ax2.imshow(result)  # the imaginary part is an artifact
    plt.show()
