from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage as scp

from spfluo.utils.array import array_namespace

if TYPE_CHECKING:
    from spfluo.utils.array import Array


def threshold(im, thresh_val):
    im_out = (im >= thresh_val) * im
    return im_out


def crop_center(image, size):
    """crop a sub array of shape 'size' at the center of the image.
    It pads the images at the dimensions where the shape is smaller than
    the crop size"""
    nb_dim = len(image.shape)
    padding = []
    for d in range(nb_dim):
        if image.shape[d] < size[d]:
            padding_d = int((size[d] - image.shape[d]) // 2 + 1)
            padding.append((padding_d, padding_d))
        else:
            padding.append((0, 0))
    padding = tuple(padding)
    image = np.pad(image, padding, mode="constant")
    size = np.array(size)
    c = [image.shape[0] // 2, image.shape[1] // 2, image.shape[2] // 2]
    cropped = image[
        c[0] - size[0] // 2 : c[0] + size[0] // 2,
        c[1] - size[1] // 2 : c[1] + size[1] // 2,
        c[2] - size[2] // 2 : c[2] + size[2] // 2,
    ]
    return cropped


def normalize(arr: "Array", min: float = 0, max: float = 1):
    xp = array_namespace(arr)
    m, M = xp.min(arr), xp.max(arr)
    if M > m:
        norm_0_1 = (arr - m) / (M - m)
    else:
        norm_0_1 = xp.zeros_like(arr)
    norm = (max - min) * norm_0_1 + min
    return norm


def resize(in_array, desired_shape):
    """zoom an image to a desired shape"""
    array_shape = np.array(list(in_array.shape))
    desired_shape = np.array(desired_shape)
    return scp.zoom(in_array, desired_shape / array_shape, order=5)


def resize_to_given_size_of_pixels(volume, pixel_size_before, pixel_size_after):
    pixel_size_before, pixel_size_after = (
        np.array(pixel_size_before),
        np.array(pixel_size_after),
    )
    shape_before = volume.shape
    ratio_shape = pixel_size_before / pixel_size_after
    shape_after = shape_before * ratio_shape
    shape_after = [int(shape_after[d] + 0.5) for d in range(len(pixel_size_after))]
    return resize(volume, shape_after)


def window_fft(im, size, sig_noise):
    fft_im = np.fft.fftshift(np.fft.fftn(im))
    window_fft_im = np.random.normal(0, sig_noise, im.shape)
    c = [im.shape[0] // 2, im.shape[1] // 2, im.shape[2] // 2]
    window_fft_im[
        c[0] - size[0] // 2 : c[0] + size[0] // 2,
        c[1] - size[1] // 2 : c[1] + size[1] // 2,
        c[2] - size[2] // 2 : c[2] + size[2] // 2,
    ] = (
        crop_center(fft_im, size)
        - window_fft_im[
            c[0] - size[0] // 2 : c[0] + size[0] // 2,
            c[1] - size[1] // 2 : c[1] + size[1] // 2,
            c[2] - size[2] // 2 : c[2] + size[2] // 2,
        ]
    )
    return np.fft.ifftn(np.fft.ifftshift(window_fft_im))


def projection_z_axis(im):
    """project image along z axis"""
    return np.sum(im, axis=0)


def pad_all_channels_of_image(im, pad_vals):
    nb_channels = im.shape[2]
    shape_padded = np.shape(np.pad(im[:, :, 0], pad_vals))
    im_padded = np.zeros((*shape_padded, 3))
    for c in range(nb_channels):
        im_c_padded = np.pad(im[:, :, c], pad_vals)
        im_padded[:, :, c] = im_c_padded
    print("shape", im_padded.shape)
    return im_padded


def stopping_criteria(energies, eps, n=4, m=2):
    if len(energies) <= n:
        return False
    mean_current = np.mean(energies[-m:])
    mean_old = np.mean(energies[-n : -n + m])
    return mean_old - mean_current < eps
