from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndii
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import DTypeLike, NDArray
from scipy.ndimage import affine_transform as affine_transform_scipy
from scipy.ndimage import fourier_shift as fourier_shift_scipy
from scipy.signal.windows import tukey as tukey_scipy
from skimage.registration import (
    phase_cross_correlation as phase_cross_correlation_skimage,
)

from spfluo.utils.array import (
    array_namespace,
    get_device,
    is_array_api_obj,
    is_cupy_array,
    is_numpy_array,
    is_torch_array,
    to_numpy,
)

if TYPE_CHECKING:
    from spfluo.utils.array import Array, array_api_module


def affine_transform(
    input: "Array",
    matrix: "Array",
    offset: Union[float, Tuple[float], "Array"] = 0.0,
    output_shape: Optional[Tuple[int]] = None,
    output: Optional[Union["Array", DTypeLike]] = None,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
    *,
    batch: bool = False,
    multichannel: bool = False,
) -> "Array":
    """Apply affine transformations to an image.
    Works with multichannel images and batches.
    Supports numpy, cupy and torch inputs.
    torch only supports linear interpolation.

    Given an output image pixel index vector ``o``, the pixel value is
    determined from the input image at position
    ``xp.dot(matrix, o) + offset``.

    Args:
        input (xp.ndarray): The input array.
            torch only supports 3D inputs.
        matrix (xp.ndarray): Must be a 'real floating' dtype matrix.
            The inverse coordinate transformation matrix,
            mapping output coordinates to input coordinates. If ``ndim`` is the
            number of dimensions of ``input``, the given matrix must have one
            of the following shapes:

                - ``(N, ndim, ndim)``: the linear transformation matrix for each
                  output coordinate.
                - ``(N, ndim,)``: assume that the 2D transformation matrix is
                  diagonal, with the diagonal specified by the given value.
                - ``(N, ndim + 1, ndim + 1)``: assume that the transformation is
                  specified using homogeneous coordinates. In this case, any
                  value passed to ``offset`` is ignored.
                - ``(N, ndim, ndim + 1)``: as above, but the bottom row of a
                  homogeneous transformation matrix is always
                  ``[0, 0, ..., 1]``, and may be omitted.

        offset (float or sequence or cp.array): The offset into the array where
            the transform is applied. If a float, ``offset`` is the same for each
            axis. If a sequence, ``offset`` should contain one value for each
            axis. If a xp.array, should be of shape (N, d) where d is the number
            of axes.
        output_shape (tuple of ints): Shape tuple. One shape for all the batch.
        output (xp.ndarray or ~xp.dtype): The array in which to place the
            output, or the dtype of the returned array.
            Not implemented in torch.
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
            Only order 1 is implemented in torch.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).
            Only ``'constant'``, ``'nearest'``, ``'reflect'`` are implemented
            in torch.
        cval (scalar): Value used for points outside the boundaries of
            the input if ``mode='constant'`` or ``mode='opencv'``. Default is
            0.0.
            Only 0.0 is implemented in torch.
        prefilter (bool): Determines if the input array is prefiltered with
            ``spline_filter`` before interpolation. The default is True, which
            will create a temporary ``float64`` array of filtered values if
            ``order > 1``. If setting this to False, the output will be
            slightly blurred if ``order > 1``, unless the input is prefiltered,
            i.e. it is the result of calling ``spline_filter`` on the original
            input.
            Not implemented in torch.

        batch (bool): if True, the first dimension is a batch dimension
            default to False
        multichannel (bool): if True, the first (or second if batch=True) is
            the channel dimension

    Returns:
        xp.ndarray:
            The transformed input. Return None if output is given.
    """
    xp = array_namespace(input, matrix)
    assert xp.isdtype(
        matrix.dtype, "real floating"
    ), "matrix dtype must be a 'real floating' dtype (float32, float64)"
    has_output = False
    if is_array_api_obj(output):
        output = xp.asarray(output)
        has_output = True

    if batch is False:
        input = input[None, ...]
        matrix = matrix[None, ...]
        if has_output:
            output = output[None]
    if multichannel is False:
        input = input[:, None, ...]

    if is_torch_array(input):
        from ._torch_functions.volume import (
            affine_transform_batched_multichannel_pytorch,
        )

        func = affine_transform_batched_multichannel_pytorch
    elif is_cupy_array(input):
        from ._cupy_functions.volume import affine_transform_batched_multichannel_cupy

        func = affine_transform_batched_multichannel_cupy
    elif is_numpy_array(input):
        func = affine_transform_batched_multichannel_scipy
    else:
        try:
            input = np.asarray(input)
            matrix = np.asarray(matrix)
            offset = np.asarray(offset)
            if has_output:
                output = np.asarray(output)
        except TypeError:
            raise ValueError(f"No backend found for {xp}")
        func = affine_transform_batched_multichannel_scipy
    out = func(
        input, matrix, offset, output_shape, output, order, mode, cval, prefilter
    )
    if has_output:
        out = output
    out = xp.asarray(out)
    if multichannel is False:
        out = out[:, 0, ...]
    if batch is False:
        out = out[0, ...]
    if not has_output:
        return out


def affine_transform_batched_multichannel_scipy(
    input: NDArray,
    matrix: NDArray,
    offset: Union[float, Tuple[float], NDArray] = 0.0,
    output_shape: Optional[Tuple[int]] = None,
    output: Optional[Union[NDArray, DTypeLike]] = None,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> NDArray:
    N, C, *image_shape = input.shape
    if output_shape is None:
        output_shape = tuple(image_shape)
    return_none = False
    if output is None:
        output = np.empty((N, C) + output_shape, dtype=input.dtype)
    elif type(output) is type:
        output = np.empty((N, C) + output_shape, dtype=output)
    else:
        return_none = True
    if isinstance(offset, float) or isinstance(offset, tuple):

        def offset_gen(_):
            return offset

    else:
        assert type(offset) is np.ndarray
        if offset.ndim <= 1:
            assert offset.ndim == 0 or (offset.shape == (1,) or offset.shape == (3,))

            def offset_gen(_):
                return offset
        elif offset.ndim == 2:
            assert offset.shape[0] == N
            assert offset.shape[-1] == (1,) or offset.shape[-1] == (3,)

            def offset_gen(i):
                return offset[i]
        else:
            raise ValueError("Wrong offset size")

    for i in range(N):
        for j in range(C):
            affine_transform_scipy(
                input[i, j],
                matrix[i],
                offset_gen(i),
                output_shape,
                output[i, j],
                order,
                mode,
                cval,
                prefilter,
            )

    if return_none:
        return

    return output


def resample(
    volume: "Array",
    sampling: tuple[float],
    order: int = 1,
    batch: bool = False,
    multichannel: bool = False,
):
    xp = array_namespace(volume)
    device = get_device(volume)
    sampling = xp.asarray(sampling, device=device, dtype=xp.float64)
    in_shape = xp.asarray(volume.shape[-3:], device=device, dtype=xp.float64)
    out_shape = xp.round(in_shape * sampling)
    D, H, W = int(out_shape[0]), int(out_shape[1]), int(out_shape[2])

    # Compute transformation matrix
    input_center, output_center = (in_shape - 1) / 2, (out_shape - 1) / 2
    H_center, H_homo = (
        xp.eye(4, device=device, dtype=xp.float64),
        xp.eye(4, device=device, dtype=xp.float64),
    )
    H_center[:3, 3] = -input_center  # 1. translation to (0,0,0)
    H_homo[[0, 1, 2], [0, 1, 2]] = sampling  # 2. homothety
    H_homo[:3, 3] = output_center  # 3. translation to center of image
    #    3-2 <- 1
    mat = H_homo @ H_center

    inv_mat = xp.linalg.inv(mat)
    if batch:
        N = volume.shape[0]
        inv_mat = xp.broadcast_to(inv_mat[None], (N, 4, 4))
    return affine_transform(
        volume,
        inv_mat,
        output_shape=(D, H, W),
        batch=batch,
        multichannel=multichannel,
        order=order,
        prefilter=False,
    )


def pad(volume: "Array", pad_width: Tuple[int]):
    xp = array_namespace(volume)
    pad_width = np.asarray(pad_width)
    if pad_width.ndim == 1:
        pad_width = np.broadcast_to(np.reshape(pad_width, (-1, 1)), (volume.ndim, 2))
    assert pad_width.shape == (volume.ndim, 2)
    padded_volume = xp.zeros(
        tuple(
            [
                s + int(before) + int(after)
                for s, (before, after) in zip(volume.shape, pad_width)
            ]
        ),
        dtype=volume.dtype,
        device=get_device(volume),
    )
    padded_volume[
        tuple(
            [
                slice(before, s - after)
                for (before, after), s in zip(pad_width, padded_volume.shape)
            ]
        )
    ] = volume

    return padded_volume


def interpolate_to_size(
    volume: "Array",
    output_size: Tuple[int, int, int],
    order=1,
    batch=False,
    multichannel=False,
) -> "Array":
    """
    Used for padding. The zoom matrix will zoom-out from the image.
    """
    xp = array_namespace(volume)
    volume = xp.asarray(volume)
    D, H, W = output_size
    input_center = (
        xp.asarray(volume.shape[-3:], device=get_device(volume), dtype=xp.float64) - 1
    ) / 2
    output_center = (
        xp.asarray(output_size, device=get_device(volume), dtype=xp.float64) - 1
    ) / 2
    mat = xp.eye(4, device=get_device(volume))
    mat[:3, 3] = output_center - input_center
    inv_mat = xp.linalg.inv(mat)
    if batch:
        N = volume.shape[0]
        inv_mat = xp.broadcast_to(inv_mat[None, ...], (N, 4, 4))
    out_vol = affine_transform(
        volume,
        inv_mat,
        output_shape=(D, H, W),
        batch=batch,
        multichannel=multichannel,
        order=order,
        prefilter=False,
    )
    return out_vol


def fourier_shift_broadcasted_scipy(
    input: NDArray,
    shift: Union[float, Sequence[float], NDArray],
    n: int = -1,
    axis: int = -1,
    output: Optional[NDArray] = None,
):
    shift = np.asarray(shift)
    if shift.ndim == 0:
        shift = np.asarray([shift] * input.ndim)
    nb_spatial_dims = shift.shape[-1]
    broadcasted_shape = np.broadcast_shapes(
        input.shape[:-nb_spatial_dims], shift.shape[:-1]
    )
    image_shape = input.shape[-nb_spatial_dims:]
    input = np.broadcast_to(input, broadcasted_shape + image_shape)
    shift = np.broadcast_to(shift, broadcasted_shape + (nb_spatial_dims,))
    output = np.empty(broadcasted_shape + image_shape, dtype=input.dtype)
    for index in np.ndindex(broadcasted_shape):
        fourier_shift_scipy(
            input[index].copy(),
            shift[index],
            n,
            axis,
            output[index],
        )
    return output


def fourier_shift(
    input: "Array",
    shift: Union[float, Sequence[float], "Array"],
    n: int = -1,
    axis: int = -1,
    output: Optional["Array"] = None,
):
    """
    Multidimensional Fourier shift filter.

    The array is multiplied with the Fourier transform of a shift operation.

    Parameters
    ----------
    input : array_like
        The input array.
        If shift is an array, input and shift will be broadcasted:
            input of shape ({...}, [...])
            where [...] corresponds to the D spatial dimensions
            and {...} corresponds to the dimensions to be broadcasted
    shift : float, sequence or array_like
        The size of the box used for filtering.
        If a float, `shift` is the same for all axes. If a sequence, `shift`
        has to contain one value for each axis.
        If an array, shift will be broadcasted with the input :
            shift must be of shape ({{...}}, D)
            where {{...}} corresponds to dimensions to be broadcasted
            and D to the number of spatial dimensions
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : ndarray, optional
        If given, the result of shifting the input is placed in this array.
        None is returned in this case.
    Returns
    -------
    fourier_shift : ndarray
        The shifted input.
        If shift is an array, {...} and {{...}} are broadcasted to (...).
        The resulting shifted array has the shape ((...), [...])
    """
    if is_numpy_array(input):
        func = fourier_shift_broadcasted_scipy
    elif is_cupy_array(input):
        from ._cupy_functions.volume import fourier_shift_broadcasted_cupy

        func = fourier_shift_broadcasted_cupy
    else:
        func = _fourier_shift_broadcasted_array_api

    output = func(
        input,
        shift,
        n,
        axis,
        output,
    )
    return output


def _fourier_shift_broadcasted_array_api(
    input: "Array",
    shift: Union[float, Sequence[float], "Array"],
    n: int = -1,
    axis: int = -1,
    output: "Optional[Array]" = None,
):
    """
    Args:
        input (Array): input in the Fourier domain ({...}, [...])
            where [...] corresponds to the N spatial dimensions
            and {...} corresponds to the batched dimensions
        shift (Array): shift to apply to the input ({{...}}, N)
            where {{...}} corresponds to batched dimensions.
        n: not implemented
        axis: not implemented
        output: not implemented
    Notes:
        {...} and {{...}} are broadcasted to (...).
    Returns:
        out (Array): input shifted in the Fourier domain. Shape ((...), [...])
    """
    xp = array_namespace(input)
    if n != -1:
        raise NotImplementedError("n should be equal to -1")
    if axis != -1:
        raise NotImplementedError("axis should be equal to -1")
    if output is not None:
        raise NotImplementedError("can't store result in output. not implemented")
    device = get_device(input)
    complex_dtype = input.dtype
    assert complex_dtype in (xp.complex64, xp.complex128)
    floating_dtype = None
    if input.dtype == xp.complex128:
        floating_dtype = xp.float64
    else:
        floating_dtype = xp.float32
    shift = xp.asarray(shift, dtype=floating_dtype, device=device)
    if shift.ndim == 0:
        shift = xp.asarray([shift] * input.ndim)
    nb_spatial_dims = shift.shape[-1]
    spatial_shape = input.shape[-nb_spatial_dims:]
    shift = xp.reshape(shift, (*shift.shape[:-1], *((1,) * nb_spatial_dims), -1))

    grid_freq = xp.stack(
        xp.meshgrid(
            *[xp.asarray(np.fft.fftfreq(int(s)), device=device) for s in spatial_shape],
            indexing="ij",
        ),
        axis=-1,
    )
    phase_shift = xp.sum(grid_freq * shift, axis=-1)

    # Fourier shift
    out = input * xp.exp(
        -1j * 2 * xp.pi * xp.astype(phase_shift, complex_dtype, copy=False)
    )

    return out


def phase_cross_correlation_broadcasted_skimage(
    reference_image: "Array",
    moving_image: "Array",
    *,
    upsample_factor: int = 1,
    space: str = "real",
    disambiguate: bool = False,
    reference_mask: Optional["Array"] = None,
    moving_mask: Optional["Array"] = None,
    overlap_ratio: float = 0.3,
    normalization: str = "phase",
    nb_spatial_dims: Optional[int] = None,
):
    if nb_spatial_dims is None:
        return phase_cross_correlation_skimage(
            reference_image,
            moving_image,
            upsample_factor=upsample_factor,
            space=space,
            disambiguate=disambiguate,
            reference_mask=reference_mask,
            moving_mask=moving_mask,
            overlap_ratio=overlap_ratio,
            normalization=normalization,
        )
    broadcast = np.broadcast(reference_image, moving_image)
    reference_image, moving_image = np.broadcast_arrays(reference_image, moving_image)
    other_shape = broadcast.shape[:-nb_spatial_dims]
    shifts = [np.empty(other_shape) for _ in range(nb_spatial_dims)]
    errors = np.empty(other_shape)
    phasediffs = np.empty(other_shape)
    for index in np.ndindex(other_shape):
        ref_im, moving_im = reference_image[index], moving_image[index]
        s, e, p = phase_cross_correlation_skimage(
            ref_im,
            moving_im,
            upsample_factor=upsample_factor,
            space=space,
            disambiguate=disambiguate,
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


def phase_cross_correlation(
    reference_image: "Array",
    moving_image: "Array",
    *,
    upsample_factor: int = 1,
    space: str = "real",
    disambiguate: bool = False,
    reference_mask: Optional["Array"] = None,
    moving_mask: Optional["Array"] = None,
    overlap_ratio: float = 0.3,
    normalization: str = "phase",
    nb_spatial_dims: Optional[int] = None,
    multichannel: bool = False,
):
    """Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT [1]_.

    Parameters
    ----------
    reference_image : array
        Reference image.
    moving_image : array
        Image to register. Must be same dimensionality as
        ``reference_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel. Default is 1 (no upsampling).
        Not used if any of ``reference_mask`` or ``moving_mask`` is not None.
    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data. "real" means
        data will be FFT'd to compute the correlation, while "fourier"
        data will bypass FFT of input data. Case insensitive. Not
        used if any of ``reference_mask`` or ``moving_mask`` is not
        None.
    disambiguate : bool
        The shift returned by this function is only accurate *modulo* the
        image shape, due to the periodic nature of the Fourier transform. If
        this parameter is set to ``True``, the *real* space cross-correlation
        is computed for each possible shift, and the shift with the highest
        cross-correlation within the overlapping area is returned.
    reference_mask : ndarray
        Boolean mask for ``reference_image``. The mask should evaluate
        to ``True`` (or 1) on valid pixels. ``reference_mask`` should
        have the same shape as ``reference_image``.
    moving_mask : ndarray or None, optional
        Boolean mask for ``moving_image``. The mask should evaluate to ``True``
        (or 1) on valid pixels. ``moving_mask`` should have the same shape
        as ``moving_image``. If ``None``, ``reference_mask`` will be used.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images. Used only if one of ``reference_mask`` or
        ``moving_mask`` is not None.
    normalization : {"phase", None}
        The type of normalization to apply to the cross-correlation. This
        parameter is unused when masks (`reference_mask` and `moving_mask`) are
        supplied.
    nb_spatial_dims: int
        If your inputs are broadcastable, you must fill this param.
    multichannel: bool
        if True, reference_image and moving_image must have shape ({...}, C, [...])
        where [...] corresponds to the spatial dims

    Returns
    -------
    shift : array
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with
        the axis order of the input array.
    error : float
        Translation invariant normalized RMS error between
        ``reference_image`` and ``moving_image``.
        is "always".
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).
    """
    extra_kwargs = {}
    if multichannel:
        func = _phase_cross_correlation_broadcasted_array_api
        extra_kwargs["multichannel"] = True
    elif is_numpy_array(reference_image):
        func = phase_cross_correlation_broadcasted_skimage
    elif is_cupy_array(reference_image):
        from ._cupy_functions.volume import phase_cross_correlation_broadcasted_cucim

        func = phase_cross_correlation_broadcasted_cucim
    else:
        func = _phase_cross_correlation_broadcasted_array_api

    shift, error, phasediff = func(
        reference_image,
        moving_image,
        upsample_factor=upsample_factor,
        space=space,
        disambiguate=disambiguate,
        reference_mask=reference_mask,
        moving_mask=moving_mask,
        overlap_ratio=overlap_ratio,
        normalization=normalization,
        nb_spatial_dims=nb_spatial_dims,
        **extra_kwargs,
    )

    return shift, error, phasediff


def swap_dims(arr: "Array", axis1: int, axis2: int):
    xp = array_namespace(arr)
    if axis1 < 0:
        axis1 += arr.ndim
    if axis2 < 0:
        axis2 += arr.ndim
    axes = list(range(arr.ndim))
    i1, i2 = axes.index(axis1), axes.index(axis2)
    axes[i1], axes[i2] = axes[i2], axes[i1]
    return xp.permute_dims(arr, tuple(axes))


def _upsampled_dft(
    data: "Array",
    upsampled_region_size: int,
    upsample_factor: int = 1,
    axis_offsets: "Array|None" = None,
    nb_spatial_dims=3,
):
    xp = array_namespace(data)
    device = get_device(data)
    if axis_offsets is None:
        raise NotImplementedError()
    upsampled_region_size = [upsampled_region_size] * nb_spatial_dims
    dim_properties = list(
        zip(
            range(-nb_spatial_dims, 0),
            data.shape[-nb_spatial_dims:],
            upsampled_region_size,
            xp.permute_dims(axis_offsets, (-1, *tuple(range(axis_offsets.ndim - 1)))),
        )
    )
    im2pi = 1j * 2 * xp.pi
    for ax, n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (
            xp.arange(ups_size, device=device, dtype=xp.float32) - ax_offset[..., None]
        )[..., None] * xp.asarray(
            np.fft.fftfreq(n_items, d=upsample_factor), device=device
        )
        kernel = xp.exp(-im2pi * xp.astype(kernel, data.dtype))  # shape (..., U, s_i)
        for _ in range(nb_spatial_dims - 1):
            kernel = xp.expand_dims(kernel, axis=-3)
        kernel  # shape (..., 1, .., 1, U, s_i)
        data  # shape (..., s_1, .., s_i .., s_N)
        data = swap_dims(data, ax, -1)  # shape (..., s_1, .., s_N, .., s_i)
        data = data[..., None]  # shape (..., s_1, .., s_N, .., s_i, 1)
        data = (kernel @ data)[..., 0]  # shape (..., s_1, .., s_N, .., U)
        data = swap_dims(data, ax, -1)
    return data


def unravel_index(indices: "Array", shape: tuple[int, ...]):
    xp = array_namespace(indices)
    device = get_device(indices)
    dim_prod = np.flip(np.cumprod((1, *shape[:0:-1])))
    dim_prod = xp.asarray(dim_prod.tolist(), dtype=indices.dtype, device=device)
    return (indices[..., None] // dim_prod) % xp.asarray(
        shape, dtype=indices.dtype, device=device
    )


def _cross_correlation_multichannel_max(
    x: "Array", y: "Array", normalization: str, nb_spatial_dims: int = None
) -> "tuple[Array, Array, Array]":
    """Compute multichannel cross-correlation between x and y
    Params:
        x (Array) of shape ({...}, C, ...)
            where (...) corresponds to the N spatial dimensions
            and C is the number of channels
        y (Array) of shape ({{...}}, C, ...)
            where (...) corresponds to the N spatial dimensions
    Returns:
        maxi (Array): cross correlation maximum of shape [...]
        shift (Tuple[Array]): tuple of N tensors of size [...]
        image_product (Array): product of size ([...], ...)
    Notes
        x and y must be broadcastable
        {...} and {{...}} are broadcasted to [...]
    """
    xp = array_namespace(x, y)
    nb_spatial_dims = nb_spatial_dims if nb_spatial_dims is not None else x.ndim - 1
    output_shape = np.broadcast_shapes(x.shape, y.shape)
    other_shape, spatial_shape = (
        output_shape[: -nb_spatial_dims - 1],
        output_shape[-nb_spatial_dims:],
    )
    z = x * xp.conj(y)
    # reduce channel dim
    z = xp.sum(z, axis=-nb_spatial_dims - 1)
    if normalization == "phase":
        z_abs = xp.abs(z)
        eps = xp.finfo(z_abs.dtype).eps
        z /= xp.where(
            z_abs > 100 * eps,
            xp.asarray(z_abs, dtype=z.dtype),
            xp.asarray(100 * eps, dtype=z.dtype),
        )
    cc = xp.fft.ifftn(z, axes=tuple(range(-nb_spatial_dims, 0)))
    cc = xp.abs(cc)
    cc = xp.reshape(cc, other_shape + (-1,))
    max_idx = xp.argmax(cc, axis=-1)
    indices = tuple(
        xp.asarray(idx, dtype=max_idx.dtype, device=get_device(max_idx))
        for idx in np.ix_(*tuple(np.arange(cc.shape[i]) for i in range(cc.ndim - 1)))
    )
    maxi = cc[(*indices, max_idx)]
    shift = unravel_index(max_idx, spatial_shape)
    return maxi**2, shift, z


def _phase_cross_correlation_broadcasted_array_api(
    reference_image: "Array",
    moving_image: "Array",
    *,
    upsample_factor: int = 1,
    space: str = "real",
    disambiguate: bool = False,
    reference_mask: Optional["Array"] = None,
    moving_mask: Optional["Array"] = None,
    overlap_ratio: float = 0.3,
    normalization: str = "phase",
    nb_spatial_dims: Optional[int] = None,
    multichannel: bool = False,
) -> Tuple["Array", "Array", "Array"]:
    """Phase cross-correlation between a reference and moving_images
    Params:
        reference (Array): image of shape ({...}, (C), [...])
            where [...] corresponds to the N spatial dimensions
        moving_images (Array): images to register of shape ({{...}}, (C), [...])
            where [...] corresponds to the N spatial dimensions
        upsample_factor (float): upsampling factor.
            Images will be registered up to 1/upsample_factor.
        space: not implemented
        disambiguate: not implemented
        reference_mask: not implemented
        moving_mask: not implemented
        overlap_ratio: not implemented
        normalization : {"phase", None}
            The type of normalization to apply to the cross-correlation. This
            parameter is unused when masks (`reference_mask` and `moving_mask`) are
            supplied.
        nb_spatial_dims (int): specify the N spatial dimensions
        multichannel (bool): optional C dimension.
    Returns:
        {...} and {{...}} shapes are broadcasted to (...)
        error (Array): tensor of shape (...)
        shift (Tuple[Array]): tuple of N tensors of size (...)
    """
    xp = array_namespace(reference_image, moving_image)
    device1, device2 = get_device(reference_image), get_device(moving_image)
    if device1 != device2:
        raise ValueError(f"found args on two devices: {device1}, {device2}")
    device = device1
    if disambiguate:
        raise NotImplementedError(
            "array api masked cross correlation disambiguate is not implemented"
        )
    if reference_mask is not None or moving_mask is not None:
        raise NotImplementedError(
            "array api masked cross correlation is not implemented"
        )
    if nb_spatial_dims is None:
        assert reference_image.shape == moving_image.shape
        nb_spatial_dims = reference_image.ndim
        if multichannel:
            nb_spatial_dims -= 1
    if not multichannel:
        reference_image = xp.expand_dims(reference_image, axis=-nb_spatial_dims - 1)
        moving_image = xp.expand_dims(moving_image, axis=-nb_spatial_dims - 1)
    output_shape = np.broadcast_shapes(reference_image.shape, moving_image.shape)
    if space == "real":
        (floating_dtype,) = set((reference_image.dtype, moving_image.dtype))
        assert floating_dtype in (xp.float32, xp.float64)
        if floating_dtype == xp.float32:
            complex_dtype = xp.complex64
        else:
            complex_dtype = xp.complex128
        reference_image_freq = xp.fft.fftn(
            xp.astype(reference_image, complex_dtype),
            axes=tuple(range(-nb_spatial_dims, 0)),
        )
        moving_image_freq = xp.fft.fftn(
            xp.astype(moving_image, complex_dtype),
            axes=tuple(range(-nb_spatial_dims, 0)),
        )
    elif space == "fourier":
        (complex_dtype,) = set((reference_image.dtype, moving_image.dtype))
        reference_image_freq = xp.astype(reference_image, complex_dtype)
        moving_image_freq = xp.astype(moving_image, complex_dtype)
    other_shapes, spatial_shapes = (
        output_shape[: -nb_spatial_dims - 1],
        output_shape[-nb_spatial_dims:],
    )
    channel_shape = (output_shape[-nb_spatial_dims - 1],)
    assert other_shapes + channel_shape + spatial_shapes == output_shape
    assert len(spatial_shapes) == nb_spatial_dims
    midpoints = xp.asarray(
        [axis_size // 2 for axis_size in spatial_shapes], device=device, dtype=xp.int64
    )

    # Single pixel registration
    error, shift, image_product = _cross_correlation_multichannel_max(
        reference_image_freq,
        moving_image_freq,
        normalization,
        nb_spatial_dims=nb_spatial_dims,
    )

    # Now change shifts so that they represent relative shifts and not indices
    spatial_shapes_broadcasted = xp.asarray(
        np.broadcast_to(spatial_shapes, shift.shape), device=device
    )
    shift[shift > midpoints] -= spatial_shapes_broadcasted[shift > midpoints]

    rg00 = xp.sum(
        (reference_image_freq * xp.conj(reference_image_freq)),
        axis=tuple(
            range(
                reference_image_freq.ndim - nb_spatial_dims - 1,
                reference_image_freq.ndim,
            )
        ),
    )
    rf00 = xp.sum(
        (moving_image_freq * xp.conj(moving_image_freq)),
        axis=tuple(
            range(moving_image_freq.ndim - nb_spatial_dims - 1, moving_image_freq.ndim)
        ),
    )

    if upsample_factor == 1:
        spatial_size = xp.prod(
            xp.asarray(spatial_shapes, dtype=complex_dtype, device=device)
        )
        rg00 = rg00 / spatial_size
        rf00 = rf00 / spatial_size
    else:
        shift = xp.astype(shift, xp.float32)
        shift = xp.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = math.ceil(upsample_factor * 1.5)
        dftshift = math.trunc(upsampled_region_size / 2.0)
        sample_region_offset = dftshift - shift * upsample_factor
        cross_correlation = xp.conj(
            _upsampled_dft(
                xp.conj(image_product),
                upsampled_region_size,
                upsample_factor,
                sample_region_offset,
                nb_spatial_dims,
            )
        )
        cross_correlation = xp.abs(cross_correlation)
        cross_correlation_reshaped = xp.reshape(
            cross_correlation, (*tuple(other_shapes), -1)
        )
        max_idx = xp.argmax(cross_correlation_reshaped, axis=-1)
        indices = tuple(
            xp.asarray(idx, dtype=max_idx.dtype, device=device)
            for idx in np.ix_(
                *tuple(
                    np.arange(cross_correlation_reshaped.shape[i])
                    for i in range(cross_correlation_reshaped.ndim - 1)
                )
            )
        )
        error = cross_correlation_reshaped[(*indices, max_idx)] ** 2
        maxima = unravel_index(max_idx, cross_correlation.shape[-nb_spatial_dims:])
        maxima -= dftshift

        shift += xp.astype(maxima, xp.float32) / xp.asarray(
            upsample_factor, dtype=xp.float32, device=device
        )

    error = 1.0 - error / (xp.real(rg00) * xp.real(rf00))
    error = xp.sqrt(xp.abs(error))

    return tuple([shift[..., i] for i in range(shift.shape[-1])]), error, None


def cartesian_prod(*arrays: "Array"):
    xp = array_namespace(*arrays)
    return xp.reshape(
        xp.stack(xp.meshgrid(*arrays, indexing="ij"), axis=-1), (-1, len(arrays))
    )


def discretize_sphere_uniformly(
    xp: "array_api_module",
    N: int,
    M: int,
    symmetry: int = 1,
    product: bool = False,
    dtype=None,
    device=None,
):
    """Generates a list of the two first euler angles that describe a uniform
    discretization of the sphere with the Fibonnaci sphere algorithm.
    Params:
        xp: numpy, torch or cupy
        N, the number of axes (two first euler angles)
        M, the number of rotations around the axes (third euler angle)
            symmetry, the order of symmetry to reduce the range of the 3rd angle.
            Default to 1, no symmetry product
            If True return the cartesian product between the axes and the rotations

    Returns: (theta, phi, psi), precision
        precision, a float representing an approximation of the sampling done
        (theta, phi, psi), a tuple of 1D arrays containing the 3 euler angles
            theta.shape == phi.shape == (N,)
            psi.shape == (M,)
        if product is true,
            theta.shape == phi.shape == psi.shape == (N*M,)
    """
    epsilon = 0.5
    goldenRatio = (1 + 5**0.5) / 2
    i = xp.arange(0, N, device=device, dtype=dtype)
    theta = xp.remainder(
        2 * xp.pi * i / goldenRatio, xp.asarray(2 * xp.pi, device=device, dtype=dtype)
    )
    phi = xp.acos(1 - 2 * (i + epsilon) / N)
    psi = xp.linspace(0, 2 * np.pi / symmetry, M, device=device, dtype=dtype)
    if product:
        theta, psi2 = cartesian_prod(theta, psi).T
        phi, _ = cartesian_prod(phi, psi).T
        psi = psi2

    precision_axes: float = (
        (180 / xp.pi) * 2 * xp.pi**0.5 / N**0.5
    )  # aire autour d'un point = 4*pi/N
    precision_rot = (180 / xp.pi) * 2 * xp.pi / symmetry / M
    theta, phi, psi = theta * 180 / xp.pi, phi * 180 / xp.pi, psi * 180 / xp.pi
    return (theta, phi, psi), (precision_axes, precision_rot)


def center_of_mass(volume: "Array"):
    xp = array_namespace(volume)
    if not xp.isdtype(volume.dtype, "real floating"):
        dtype = xp.float64
        assert not xp.isdtype(volume.dtype, "complex floating")
        volume_float = xp.astype(volume, xp.float64)
    else:
        dtype = volume.dtype
        volume_float = volume
    kwargs = dict(dtype=dtype, device=get_device(volume))
    zz, yy, xx = xp.meshgrid(
        xp.arange(volume.shape[-3], **kwargs),
        xp.arange(volume.shape[-2], **kwargs),
        xp.arange(volume.shape[-1], **kwargs),
        indexing="ij",
    )
    S = xp.sum(volume_float)
    return (
        float(xp.sum(volume_float * zz) / S),
        float(xp.sum(volume_float * yy) / S),
        float(xp.sum(volume_float * xx) / S),
    )


def translate(volume: "Array", vec: "Array", order: int = 1):
    """Translate volume of vec
    Args:
        volume: shape (D, H, W)
        vec: shape (3,)
    """
    xp = array_namespace(volume, vec)
    (device,) = set((get_device(volume), get_device(vec)))
    return affine_transform(
        volume,
        xp.eye(3, device=device, dtype=vec.dtype),
        offset=-vec,
        output_shape=volume.shape,
        order=order,
    )


def move_center_of_mass_to_center(volume: "Array", order: int = 1):
    xp = array_namespace(volume)
    tvec = (xp.asarray(volume.shape) - 1) / 2 - xp.asarray(center_of_mass(volume))
    return translate(volume, xp.asarray(tvec, device=get_device(volume)), order=order)


def disp3D(*ims: np.ndarray, fig=None, axis_off=False):
    if fig is None:
        fig = plt.figure()
    axes = fig.subplots(1, len(ims))
    if len(ims) == 1:
        axes = [axes]
    for i in range(len(ims)):
        views = [
            ims[i][ims[i].shape[0] // 2, :, :],
            ims[i][:, ims[i].shape[1] // 2, :],
            ims[i][:, :, ims[i].shape[2] // 2],
        ]
        axes[i].set_aspect(1.0)
        # views = [normalize_patches(torch.from_numpy(v)).cpu().numpy() for v in views]

        divider = make_axes_locatable(axes[i])
        # below height and pad are in inches

        ax_x = divider.append_axes(
            "right",
            size=1,
            pad=0,
            sharey=axes[i],
        )
        ax_y = divider.append_axes(
            "bottom",
            size=1,
            pad=0,
            sharex=axes[i],
        )

        # make some labels invisible
        axes[i].xaxis.set_tick_params(
            labeltop=True, top=True, labelbottom=False, bottom=False
        )
        ax_x.yaxis.set_tick_params(labelleft=False, left=False, right=True)
        ax_x.xaxis.set_tick_params(
            top=True, labeltop=True, bottom=True, labelbottom=False
        )
        ax_y.xaxis.set_tick_params(bottom=True, labelbottom=False, top=False)
        ax_y.yaxis.set_tick_params(right=True)

        # show slice info
        if not axis_off:
            axes[i].text(
                0,
                2,
                f"Z={ims[i].shape[0]//2}",
                color="white",
                bbox=dict(boxstyle="square"),
            )
            ax_x.text(
                0,
                2,
                f"Y={ims[i].shape[1]//2}",
                color="white",
                bbox=dict(boxstyle="square"),
            )
            ax_y.text(
                0,
                2,
                f"X={ims[i].shape[2]//2}",
                color="white",
                bbox=dict(boxstyle="square"),
            )
        vmin = min(views[0].min(), views[1].min(), views[2].min())
        vmax = max(views[0].max(), views[1].max(), views[2].max())
        axes[i].imshow(views[0], cmap="gray", vmin=vmin, vmax=vmax)
        ax_y.imshow(views[1], cmap="gray", vmin=vmin, vmax=vmax)
        ax_x.imshow(ndii.rotate(views[2], 90)[::-1], cmap="gray", vmin=vmin, vmax=vmax)
        axes[i].set_xlim(0, ims[i].shape[2] - 1)
        axes[i].set_ylim(0, ims[i].shape[1] - 1)
        axes[i].yaxis.set_inverted(True)

    if axis_off:
        for ax in axes:
            ax.set_axis_off()
    return fig, axes


def disp2D(*ims, fig=None, **imshowkwargs):
    if fig is None:
        fig = plt.figure()
    h = int(math.floor(len(ims) ** 0.5))
    w = int(math.ceil(len(ims) / h))
    axes = fig.subplots(h, w)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
        for ax in axes:
            ax.set_axis_off()
        for i in range(len(ims)):
            axes[i].imshow(ims[i], **imshowkwargs)
    else:
        axes.set_axis_off()
        axes.imshow(ims[0], **imshowkwargs)
    return fig, axes


def disp2D_compare(fig, *ims, **imshowkwargs):
    h = int(np.floor(len(ims) ** 0.5))
    w = int(np.ceil(len(ims) / h))
    axes = fig.subplots(h, w)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
        for ax in axes:
            ax.set_axis_off()
        for i in range(len(ims)):
            im = np.concatenate(tuple(ims[i]), axis=1)
            axes[i].imshow(im, **imshowkwargs)
    else:
        axes.set_axis_off()
        im = np.concatenate(tuple(ims[0]), axis=1)
        axes.imshow(im, **imshowkwargs)


def get_random_3d_vector(norm=None):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0, np.pi * 2)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    if norm is None:
        norm = 1
    if norm < 0:
        norm = 0
    return norm * np.array([x, y, z])


def get_surfaces(self, corners_points):
    p = corners_points
    return np.array(
        [
            [p[0], p[1], p[3]],
            [p[0], p[4], p[3]],
            [p[0], p[4], p[1]],
            [p[6], p[7], p[2]],
            [p[6], p[2], p[5]],
            [p[6], p[7], p[5]],
        ]
    )


def get_plane_equation(self, s):
    normal1 = np.cross(s[1] - s[0], s[2] - s[0])
    normal1 /= np.linalg.norm(normal1)
    return np.concatenate([normal1, [-np.dot(normal1, s[0])]])


def get_planes_intersection(self, s1, s2):
    """tested"""
    equation1 = self.get_plane_equation(s1)
    equation2 = self.get_plane_equation(s2)

    vec1, vec2 = equation1[:3], equation2[:3]
    line_vec = np.cross(vec1, vec2)
    A = np.array([vec1, vec2, line_vec])
    d = np.array([-equation1[3], -equation2[3], 0.0]).reshape(3, 1)

    if np.linalg.det(A) == 0:
        return False, None
    else:
        p_inter = np.linalg.solve(A, d).T
        return True, (line_vec, p_inter[0])


def get_lines_intersection(self, eq1, eq2):
    A = np.array([eq1[:2], eq2[:2]])
    d = -np.array([eq1[2], eq2[2]])
    if np.linalg.det(A) == 0:
        return False, None
    else:
        p_inter = np.linalg.solve(A, d).T
        return True, p_inter


def line_crossing_segment(self, line, segment):
    """tested"""

    vec_line, p_line = line
    vec_segment = segment[1] - segment[0]
    A = np.array([vec_segment, -vec_line]).T
    d = (p_line - segment[0]).reshape(2, 1)
    if np.linalg.det(A) == 0:
        return False, None
    t1, t2 = np.linalg.solve(A, d).reshape(-1)
    return t1 >= 0 and t1 <= 1, t2


def line_intersect_surface(self, line, surface):
    """tested"""
    plane_basis = np.array([surface[1] - surface[0], surface[2] - surface[0]])
    plane_basis_position = surface[0]
    plane_orthonormal_basis = plane_basis / np.linalg.norm(plane_basis, axis=1)[:, None]

    def projector(x):
        return np.array(
            [
                np.dot(plane_orthonormal_basis[0], x - plane_basis_position),
                np.dot(plane_orthonormal_basis[1], x - plane_basis_position),
            ]
        )

    projected_line = (
        projector(line[1] + 10 * line[0]) - projector(line[1]),
        projector(line[1]),
    )
    projected_surface = np.array(
        [projector(surface[0]), projector(surface[1]), projector(surface[2])]
    )
    p4 = projected_surface[1] + projected_surface[2]
    segments = np.array(
        [
            [projected_surface[0], projected_surface[1]],
            [projected_surface[0], projected_surface[2]],
            [projected_surface[1], p4],
            [projected_surface[2], p4],
        ]
    )

    out = [self.line_crossing_segment(projected_line, seg) for seg in segments]
    t = list(map(lambda x: x[1], filter(lambda x: x[0], out)))
    if len(t) > 0:
        return True, (min(t), max(t))
    else:
        return False, (None, None)


def surfaces_intersect(self, s1, s2):
    ret, line = self.get_planes_intersection(s1, s2)
    if not ret:
        return False
    ret1, (tmin1, tmax1) = self.line_intersect_surface(line, s1)
    ret2, (tmin2, tmax2) = self.line_intersect_surface(line, s2)
    if ret1 and ret2:
        return ((tmin1 <= tmax2) and (tmin2 <= tmin1)) or (
            (tmin2 <= tmax1) and (tmin1 <= tmin2)
        )
    else:
        return False


def pointcloud_intersect(self, corners1, corners2):
    surfaces1 = self.get_surfaces(corners1)
    surfaces2 = self.get_surfaces(corners2)
    intersections = [
        self.surfaces_intersect(s1, s2)
        for s1, s2 in itertools.product(surfaces1, surfaces2)
    ]
    return any(intersections)


def are_volumes_aligned(vol1, vol2, atol=0.1, nb_spatial_dims=3):
    (dz, dy, dx), _, _ = phase_cross_correlation(
        vol1,
        vol2,
        upsample_factor=10,
        disambiguate=True,
        normalization=None,
        nb_spatial_dims=nb_spatial_dims,
    )
    n = (dz**2 + dy**2 + dx**2) ** 0.5
    return n <= atol


def assert_volumes_translated(vol1, vol2, atol=0.1, nb_spatial_dims=3):
    """check if vol1 is a translation of vol2"""
    _, error, _ = phase_cross_correlation(
        vol1,
        vol2,
        upsample_factor=10,
        disambiguate=True,
        normalization=None,
        nb_spatial_dims=nb_spatial_dims,
    )
    np.testing.assert_allclose(to_numpy(error), 0, atol=atol)


def tukey(
    xp: "array_api_module", shape: tuple[int], alpha: float = 0.5, sym: bool = True
):
    tukeys = [tukey_scipy(s, alpha=alpha, sym=sym) for s in shape]
    tukeys_reshaped = [
        xp.reshape(t, (1,) * i + (-1,) + (1,) * (len(shape) - i - 1))
        for i, t in enumerate(tukeys)
    ]
    final_window = tukeys_reshaped[0]
    for t in tukeys_reshaped[1:]:
        final_window = final_window * t
    return xp.asarray(final_window)
