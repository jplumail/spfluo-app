from typing import Optional, Sequence, Tuple, Union

import cupy as cp
from cucim.skimage.registration import (
    phase_cross_correlation as phase_cross_correlation_cucim,
)
from cupy.typing import DTypeLike
from cupy.typing import NDArray as CPArray
from cupyx.scipy.ndimage import affine_transform as affine_transform_cupy
from cupyx.scipy.ndimage import fourier_shift as fourier_shift_cupy


def affine_transform_batched_multichannel_cupy(
    input: CPArray,
    matrix: CPArray,
    offset: Union[float, Tuple[float], CPArray] = 0.0,
    output_shape: Optional[Tuple[int]] = None,
    output: Optional[Union[CPArray, DTypeLike]] = None,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> CPArray:
    N, C, *image_shape = input.shape
    if output_shape is None:
        output_shape = tuple(image_shape)
    return_none = False
    if output is None:
        output = cp.empty((N, C) + output_shape, dtype=input.dtype)
    elif type(output) is type:
        output = cp.empty((N, C) + output_shape, dtype=output)
    else:
        return_none = True
    if isinstance(offset, float) or isinstance(offset, tuple):

        def offset_gen(_):
            return offset

    else:
        assert type(offset) is cp.ndarray

        def offset_gen(i):
            return offset[i]

    for i in range(N):
        for j in range(C):
            affine_transform_cupy(
                input[i, j],
                matrix[i],
                offset_gen(i),
                output_shape,
                output[i, j],
                order,
                mode,
                cval,
                prefilter,
                texture_memory=False,
            )

    if return_none:
        return

    return output


def fourier_shift_broadcasted_cupy(
    input: CPArray,
    shift: Union[float, Sequence[float]],
    n: int = -1,
    axis: int = -1,
    output: Optional[CPArray] = None,
):
    shift = cp.asarray(shift)
    if shift.ndim == 0:
        shift = cp.asarray([shift] * input.ndim)
    nb_spatial_dims = shift.shape[-1]
    broadcasted_shape = cp.broadcast_shapes(
        input.shape[:-nb_spatial_dims], shift.shape[:-1]
    )
    image_shape = input.shape[-nb_spatial_dims:]
    input = cp.broadcast_to(input, broadcasted_shape + image_shape)
    shift = cp.broadcast_to(shift, broadcasted_shape + (nb_spatial_dims,))
    output = cp.empty(broadcasted_shape + image_shape, dtype=input.dtype)
    for index in cp.ndindex(broadcasted_shape):
        fourier_shift_cupy(
            input[index].copy(),
            shift[index],
            n,
            axis,
            output[index],
        )
    return output


def phase_cross_correlation_broadcasted_cucim(
    reference_image: CPArray,
    moving_image: CPArray,
    *,
    upsample_factor: int = 1,
    space: str = "real",
    disambiguate: bool = False,
    reference_mask: Optional[CPArray] = None,
    moving_mask: Optional[CPArray] = None,
    overlap_ratio: float = 0.3,
    normalization: str = "phase",
    nb_spatial_dims: Optional[int] = None,
):
    if nb_spatial_dims is None:
        shifts, error, phasediff = phase_cross_correlation_cucim(
            reference_image,
            moving_image,
            upsample_factor=upsample_factor,
            space=space,
            disambiguate=disambiguate,
            return_error="always",
            reference_mask=reference_mask,
            moving_mask=moving_mask,
            overlap_ratio=overlap_ratio,
            normalization=normalization,
        )
        shifts = tuple([cp.array(s, dtype=float) for s in shifts])
        return shifts, error, phasediff

    broadcast = cp.broadcast(reference_image, moving_image)
    reference_image, moving_image = cp.broadcast_arrays(reference_image, moving_image)
    other_shape = broadcast.shape[:-nb_spatial_dims]
    shifts = [cp.empty(other_shape) for _ in range(nb_spatial_dims)]
    errors = cp.empty(other_shape)
    phasediffs = cp.empty(other_shape)
    for index in cp.ndindex(other_shape):
        ref_im, moving_im = reference_image[index], moving_image[index]
        s, e, p = phase_cross_correlation_cucim(
            ref_im,
            moving_im,
            upsample_factor=upsample_factor,
            space=space,
            disambiguate=disambiguate,
            return_error="always",
            reference_mask=reference_mask,
            moving_mask=moving_mask,
            overlap_ratio=overlap_ratio,
            normalization=normalization,
        )
        for i in range(nb_spatial_dims):
            shifts[i][index] = s[i]
        errors[index] = e
        phasediffs[index] = p
    return tuple(shifts), errors, phasediffs
