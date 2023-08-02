import itertools
from typing import Optional, Sequence, Tuple, Union

import array_api_compat.cupy
import array_api_compat.numpy
import array_api_compat.torch
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndii
import torch
import torch.nn.functional as F
from cucim.skimage.registration import (
    phase_cross_correlation as phase_cross_correlation_cucim,
)
from cupy.typing import NDArray as CPArray
from cupyx.scipy.ndimage import affine_transform as affine_transform_cupy
from cupyx.scipy.ndimage import fourier_shift as fourier_shift_cupy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import DTypeLike, NDArray
from scipy.ndimage import affine_transform as affine_transform_scipy
from scipy.ndimage import fourier_shift as fourier_shift_scipy
from skimage.registration import (
    phase_cross_correlation as phase_cross_correlation_skimage,
)

from spfluo.utils.transform import get_zoom_matrix

from ._array import Array, array_api_compat
from .memory import split_batch_func


def affine_transform(
    input: Array,
    matrix: Array,
    offset: Union[float, Tuple[float], Array] = 0.0,
    output_shape: Optional[Tuple[int]] = None,
    output: Optional[Union[Array, DTypeLike]] = None,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
    *,
    batch: bool = False,
    multichannel: bool = False,
) -> Array:
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
        matrix (xp.ndarray): The inverse coordinate transformation matrix,
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
    xp = array_api_compat.array_namespace(input, matrix)
    has_output = False
    if array_api_compat.is_array_api_obj(output):
        output = xp.asarray(output)
        has_output = True

    if batch is False:
        input = input[None]
        matrix = matrix[None]
        if has_output:
            output = output[None]
    if multichannel is False:
        input = input[:, None]

    if xp == array_api_compat.torch:
        func = affine_transform_batched_multichannel_pytorch
    elif xp == array_api_compat.cupy:
        func = affine_transform_batched_multichannel_cupy
    elif xp == array_api_compat.numpy:
        func = affine_transform_batched_multichannel_scipy
    else:
        raise ValueError(f"No backend found for {xp}")
    out = func(
        input, matrix, offset, output_shape, output, order, mode, cval, prefilter
    )
    if has_output:
        out = output
    if multichannel is False:
        out = out[:, 0]
    if batch is False:
        out = out[0]
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
    if type(offset) is float or type(offset) is tuple:

        def offset_gen(_):
            return offset

    else:
        assert type(offset) is np.ndarray

        def offset_gen(i):
            return offset[i]

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
    if type(offset) is float or type(offset) is tuple:

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


def affine_transform_batched_multichannel_pytorch(
    input: torch.Tensor,
    matrix: torch.Tensor,
    offset=0.0,
    output_shape=None,
    output=None,
    order=1,
    mode="zeros",
    cval=0.0,
    prefilter=True,
) -> torch.Tensor:
    """Rotate the volume according to the transform matrix.
    Matches the `scipy.ndimage.affine_transform` function at best with the
    `torch.nn.functional.grid_sample` function

    Args:
        input (torch.Tensor): 3D images of shape (N, C, D, H, W)
        matrix (torch.Tensor)
            transform matrices of shape (N, 3), (N,3,3), (N,4,4) or (N,3,4).
        offset (float or torch.Tensor): offset of the grid.
        output_shape (tuple): shape of the output.
        output: not implemented
        order (int): must be 1. Only linear interpolation is implemented.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode
            Only ``'constant'``, ``'nearest'``, ``'reflect'`` are implemented.
        cval (float): cannot be different than 0.0
        prefilter (bool): not implemented

    Returns:
        torch.Tensor: Rotated volumes of shape (N, C, D, H, W)
    """
    N, C, D, H, W = input.size()
    tensor_kwargs = dict(device=input.device, dtype=input.dtype)

    if type(offset) == float:
        tvec = torch.tensor([offset, offset, offset], **tensor_kwargs).expand(N, 3)
    elif offset.shape == (3,):
        tvec = torch.as_tensor(offset, **tensor_kwargs).expand(N, 3)
    elif offset.shape == (N, 3):
        tvec = torch.as_tensor(offset, **tensor_kwargs)
    else:
        raise ValueError(
            "Offset should be a float, a sequence of size 3 or a tensor of size (N,3)."
        )

    if matrix.size() == torch.Size([N, 3, 3]):
        rotMat = matrix
    elif matrix.size() == torch.Size([N, 3]):
        rotMat = torch.stack([torch.diag(matrix[i]) for i in range(N)])
    elif matrix.size() == torch.Size([N, 4, 4]) or matrix.size() == torch.Size(
        [N, 3, 4]
    ):
        rotMat = matrix[:, :3, :3]
        tvec = matrix[:, :3, 3]
    else:
        raise ValueError(
            "Matrix should be a tensor of shape"
            f"{(N,3)}, {(N,3,3)}, {(N,4,4)} or {(N,3,4)}."
            f"Found matrix of shape {matrix.size()}"
        )

    if output_shape is None:
        output_shape = (D, H, W)

    if output is not None:
        raise NotImplementedError()

    if order > 1:
        raise NotImplementedError()

    pytorch_modes = {"constant": "zeros", "nearest": "border", "reflect": "reflection"}
    if mode not in pytorch_modes:
        raise NotImplementedError(f"Only {pytorch_modes.keys()} are available")
    pt_mode = pytorch_modes[mode]

    if cval != 0:
        raise NotImplementedError()

    if prefilter:
        raise NotImplementedError()

    return _affine_transform(
        input, rotMat, tvec, output_shape, pt_mode, **tensor_kwargs
    )


def _affine_transform(input, rotMat, tvec, output_shape, mode, **tensor_kwargs):
    output_shape = list(output_shape)
    rotated_vol = input.new_empty(list(input.shape[:2]) + output_shape)
    grid = torch.stack(
        torch.meshgrid(
            [torch.linspace(0, d - 1, steps=d, **tensor_kwargs) for d in output_shape],
            indexing="ij",
        ),
        dim=-1,
    )
    c = torch.tensor([0 for d in output_shape], **tensor_kwargs)
    input_shape = torch.as_tensor(input.shape[2:], **tensor_kwargs)
    for start, end in split_batch_func("affine_transform", input, rotMat):
        grid_batch = (
            (rotMat[start:end, None, None, None] @ ((grid - c)[None, ..., None]))[
                ..., 0
            ]
            + c
            + tvec[start:end, None, None, None, :]
        )
        grid_batch = -1 + 1 / input_shape + 2 * grid_batch / input_shape
        rotated_vol[start:end] = F.grid_sample(
            input[start:end],
            grid_batch[:, :, :, :, [2, 1, 0]],
            mode="bilinear",
            align_corners=False,
            padding_mode=mode,
        )
    rotated_vol[:, :, : output_shape[0], : output_shape[1], : output_shape[2]]
    return rotated_vol


def fftn(x: torch.Tensor, dim: Tuple[int] = None, out=None) -> torch.Tensor:
    """Computes N dimensional FFT of x in batch. Tries to avoid out-of-memory errors.

    Args:
        x: data
        dim: tuple of size N, dimensions where FFTs will be computed
        out: the output tensor
    Returns:
        y: data in the Fourier domain, shape of x
    """
    if dim is None or len(dim) == x.ndim:
        return torch.fft.fftn(x, out=out)
    else:
        if x.is_complex():
            dtype = x.dtype
        else:
            if x.dtype is torch.float32:
                dtype = torch.complex64
            elif x.dtype is torch.float64:
                dtype = torch.complex128
        batch_indices = torch.ones((x.ndim,), dtype=bool)
        batch_indices[list(dim)] = False
        batch_indices = batch_indices.nonzero()[:, 0]
        batch_slices = [slice(None, None) for i in range(x.ndim)]

        if out is not None:
            y = out
        else:
            y = x.new_empty(size=x.size(), dtype=dtype)
        for batch_idx in split_batch_func("fftn", x, dim):
            if type(batch_idx) is tuple:
                batch_idx = [batch_idx]
            for d, (start, end) in zip(batch_indices, batch_idx):
                batch_slices[d] = slice(start, end)
            torch.fft.fftn(x[tuple(batch_slices)], dim=dim, out=y[tuple(batch_slices)])
        return y


def ifftn(x: torch.Tensor, dim: Tuple[int] = None, out=None) -> torch.Tensor:
    """Computes N dimensional inverse FFT of x in batch.
    Tries to avoid out-of-memory errors.

    Args:
        x: data
        dim: tuple of size N, dimensions where FFTs will be computed
        out: the output tensor
    Returns:
        y: data in the Fourier domain, shape of x
    """
    if dim is None or len(dim) == x.ndim:
        return torch.fft.ifftn(x, out=out)
    else:
        if x.is_complex():
            dtype = x.dtype
        else:
            if x.dtype is torch.float32:
                dtype = torch.complex64
            elif x.dtype is torch.float64:
                dtype = torch.complex128
        batch_indices = torch.ones((x.ndim,), dtype=bool)
        batch_indices[list(dim)] = False
        batch_indices = batch_indices.nonzero()[:, 0]
        batch_slices = [slice(None, None) for i in range(x.ndim)]

        if out is not None:
            y = out
        else:
            y = x.new_empty(size=x.size(), dtype=dtype)
        for batch_idx in split_batch_func("fftn", x, dim):
            if type(batch_idx) is tuple:
                batch_idx = [batch_idx]
            for d, (start, end) in zip(batch_indices, batch_idx):
                batch_slices[d] = slice(start, end)
            torch.fft.ifftn(x[tuple(batch_slices)], dim=dim, out=y[tuple(batch_slices)])
        return y


def pad_to_size(volume: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
    output_size = torch.as_tensor(output_size)
    pad_size = torch.ceil((output_size - torch.as_tensor(volume.size())) / 2)

    padding = tuple(
        np.asarray(
            [[max(pad_size[i], 0), max(pad_size[i], 0)] for i in range(len(pad_size))],
            dtype=int,
        )
        .flatten()[::-1]
        .tolist()
    )
    output_volume = F.pad(volume, padding)

    shift = (torch.as_tensor(output_volume.size()) - output_size) / 2
    slices = [
        slice(int(np.ceil(shift[i])), -int(np.floor(shift[i])))
        if shift[i] > 0 and np.floor(shift[i]) > 0
        else slice(int(np.ceil(shift[i])), None)
        if shift[i] > 0
        else slice(None, None)
        for i in range(len(shift))
    ]

    return output_volume[tuple(slices)]


def interpolate_to_size(
    volume: Array,
    output_size: Tuple[int, int, int],
    order=1,
    batch=False,
    multichannel=False,
) -> Array:
    xp = array_api_compat.array_namespace(volume)
    volume = xp.asarray(volume)
    d, h, w = volume.shape[-3:]
    D, H, W = output_size
    mat = get_zoom_matrix(
        (d, h, w), (D, H, W), xp, device=xp.device(volume), dtype=volume.dtype
    )
    inv_mat = xp.linalg.inv(mat)
    if batch:
        N = volume.shape[0]
        inv_mat = xp.broadcast_to(inv_mat[None], (N, 4, 4))
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


def fourier_shift_broadcasted_pytorch(
    input: torch.Tensor,
    shift: Union[float, Sequence[float], torch.Tensor],
    n: int = -1,
    axis: int = -1,
    output: Optional[torch.Tensor] = None,
):
    """
    Args:
        input (torch.Tensor): input in the Fourier domain ({...}, [...])
            where [...] corresponds to the N spatial dimensions
            and {...} corresponds to the batched dimensions
        shift (torch.Tensor): shift to apply to the input ({{...}}, N)
            where {{...}} corresponds to batched dimensions.
        n: not implemented
        axis: not implemented
        output: not implemented
    Notes:
        {...} and {{...}} are broadcasted to (...).
    Returns:
        out (torch.Tensor): input shifted in the Fourier domain. Shape ((...), [...])
    """
    if n != -1:
        raise NotImplementedError("n should be equal to -1")
    if axis != -1:
        raise NotImplementedError("axis should be equal to -1")
    if output is not None:
        raise NotImplementedError("can't store result in output. not implemented")
    tensor_kwargs = {"device": input.device}
    if input.dtype == torch.complex128:
        tensor_kwargs["dtype"] = torch.float64
    elif input.dtype == torch.complex64:
        tensor_kwargs["dtype"] = torch.float32
    elif input.dtype == torch.complex32:
        tensor_kwargs["dtype"] = torch.float16
    else:
        print("Volume must be complex")
    shift = torch.asarray(shift, **tensor_kwargs)
    if shift.ndim == 0:
        shift = np.asarray([shift] * input.ndim)
    nb_spatial_dims = shift.shape[-1]
    spatial_shape = torch.as_tensor(input.size()[-nb_spatial_dims:], **tensor_kwargs)
    shift = shift.view(*shift.shape[:-1], *[1 for _ in range(nb_spatial_dims)], -1)

    grid_freq = torch.stack(
        torch.meshgrid(
            *[torch.fft.fftfreq(int(s), **tensor_kwargs) for s in spatial_shape],
            indexing="ij",
        ),
        dim=-1,
    )
    phase_shift = (grid_freq * shift).sum(-1)

    # Fourier shift
    out = input * torch.exp(-1j * 2 * torch.pi * phase_shift)

    return out


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


def fourier_shift_broadcasted_cupy(
    input: cp.ndarray,
    shift: Union[float, Sequence[float]],
    n: int = -1,
    axis: int = -1,
    output: Optional[cp.ndarray] = None,
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


def fourier_shift(
    input: Array,
    shift: Union[float, Sequence[float], Array],
    n: int = -1,
    axis: int = -1,
    output: Optional[Array] = None,
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
    xp = array_api_compat.array_namespace(input)
    if xp == array_api_compat.torch:
        func = fourier_shift_broadcasted_pytorch
    elif xp == array_api_compat.numpy:
        func = fourier_shift_broadcasted_scipy
    elif xp == array_api_compat.cupy:
        func = fourier_shift_broadcasted_cupy

    output = func(
        input,
        shift,
        n,
        axis,
        output,
    )
    return output


def hann_window(shape: Tuple[int], **kwargs) -> torch.Tensor:
    """Computes N dimensional Hann window.

    Args:
        shape: shape of the final window
        kwargs: keyword arguments for torch.hann_window function
    Returns:
         Hann window of the shape asked
    """
    windows = [torch.hann_window(s, **kwargs) for s in shape]
    view = [1] * len(shape)
    hw = torch.ones(shape, device=windows[0].device, dtype=windows[0].dtype)
    for i in range(len(windows)):
        view_ = list(view)
        view_[i] = -1
        hw *= windows[i].view(tuple(view_))
    return hw


def _upsampled_dft(
    data, upsampled_region_size, upsample_factor=1, axis_offsets=None, nb_spatial_dims=3
):
    tensor_kwargs = {"device": data.device, "dtype": None}
    upsampled_region_size = [
        upsampled_region_size,
    ] * nb_spatial_dims
    dim_properties = list(
        zip(
            data.shape[-nb_spatial_dims:],
            upsampled_region_size,
            axis_offsets.permute(-1, *tuple(range(axis_offsets.ndim - 1))),
        )
    )
    im2pi = 1j * 2 * np.pi
    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (torch.arange(ups_size, **tensor_kwargs) - ax_offset[..., None])[
            ..., None
        ] * torch.fft.fftfreq(n_items, upsample_factor, **tensor_kwargs)
        kernel = torch.exp(-im2pi * kernel)
        kernel = kernel.type(data.dtype)
        data = torch.einsum(
            kernel,
            [..., 0, nb_spatial_dims],
            data,
            [...] + list(range(1, 1 + nb_spatial_dims)),
            [..., 0] + list(range(1, 1 + nb_spatial_dims - 1)),
        )
    return data


def unravel_index(x, dims):
    one = torch.tensor([1], dtype=dims.dtype, device=dims.device)
    a = torch.cat((one, dims.flip([0])[:-1]))
    dim_prod = torch.cumprod(a, dim=0).flip([0]).type(torch.float)
    return torch.floor(x[..., None] / dim_prod) % dims


def cross_correlation_max(
    x: torch.Tensor, y: torch.Tensor, normalization: str, nb_spatial_dims: int = None
) -> Tuple[torch.Tensor]:
    """Compute cross-correlation between x and y
    Params:
        x (torch.Tensor) of shape (B, ...)
            where (...) corresponds to the N spatial dimensions
        y (torch.Tensor) of the same shape
    Returns:
        maxi (torch.Tensor): cross correlatio maximum of shape (B,)
        shift (Tuple[torch.Tensor]): tuple of N tensors of size (B,)
        image_product (torch.Tensor): product of size (B, ...)
    """
    nb_spatial_dims = nb_spatial_dims if nb_spatial_dims is not None else x.ndim
    output_shape = torch.as_tensor(
        torch.broadcast_shapes(x.size(), y.size()), dtype=torch.int64, device=x.device
    )
    spatial_dims = list(range(len(output_shape) - nb_spatial_dims, len(output_shape)))
    spatial_shape = output_shape[-nb_spatial_dims:]
    z = x * y.conj()
    if normalization == "phase":
        eps = torch.finfo(z.real.dtype).eps
        z /= torch.max(z.abs(), torch.as_tensor(100 * eps))
    cc = ifftn(z, dim=spatial_dims)
    cc = torch.mul(cc, cc.conj(), out=cc).real
    cc = torch.flatten(cc, start_dim=-nb_spatial_dims)
    maxi, max_idx = torch.max(cc, dim=-1)
    shift = unravel_index(max_idx.type(torch.int64), spatial_shape)
    return maxi, shift, z


def phase_cross_correlation_broadcasted_pytorch(
    reference_image: Array,
    moving_image: Array,
    *,
    upsample_factor: int = 1,
    space: str = "real",
    disambiguate: bool = False,
    reference_mask: Optional[Array] = None,
    moving_mask: Optional[Array] = None,
    overlap_ratio: float = 0.3,
    normalization: str = "phase",
    nb_spatial_dims: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase cross-correlation between a reference and moving_images
    Params:
        reference (torch.Tensor): image of shape ({...}, [...])
            where [...] corresponds to the N spatial dimensions
        moving_images (torch.Tensor): images to register of shape ({{...}}, [...])
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
    Returns:
        {...} and {{...}} shapes are broadcasted to (...)
        error (torch.Tensor): tensor of shape (...)
        shift (Tuple[torch.Tensor]): tuple of N tensors of size (...)
    """
    if space == "real":
        raise NotImplementedError("Space should be 'fourier'")
    if disambiguate:
        raise NotImplementedError(
            "pytorch masked cross correlation disambiguate is not implemented"
        )
    if reference_mask is not None or moving_mask is not None:
        raise NotImplementedError("pytorch masked cross correlation is not implemented")
    device = reference_image.device
    output_shape = torch.as_tensor(
        torch.broadcast_shapes(reference_image.size(), moving_image.size())
    )
    if nb_spatial_dims is None:
        nb_spatial_dims = len(output_shape)
    spatial_dims = list(range(len(output_shape) - nb_spatial_dims, len(output_shape)))
    other_dims = list(range(0, len(output_shape) - nb_spatial_dims))
    spatial_shapes = output_shape[spatial_dims]
    other_shapes = output_shape[other_dims]
    midpoints = torch.tensor(
        [torch.fix(axis_size / 2) for axis_size in spatial_shapes], device=device
    )

    # Single pixel registration
    error, shift, image_product = cross_correlation_max(
        reference_image, moving_image, normalization, nb_spatial_dims=nb_spatial_dims
    )

    # Now change shifts so that they represent relative shifts and not indices
    spatial_shapes_broadcasted = torch.broadcast_to(spatial_shapes, shift.size()).to(
        device
    )
    shift[shift > midpoints] -= spatial_shapes_broadcasted[shift > midpoints]

    spatial_size = torch.prod(spatial_shapes).type(reference_image.dtype)

    if upsample_factor == 1:
        rg00 = (
            torch.sum(
                (reference_image * reference_image.conj()),
                dim=tuple(
                    range(reference_image.ndim - nb_spatial_dims, reference_image.ndim)
                ),
            )
            / spatial_size
        )
        rf00 = (
            torch.sum(
                (moving_image * moving_image.conj()),
                dim=tuple(
                    range(moving_image.ndim - nb_spatial_dims, moving_image.ndim)
                ),
            )
            / spatial_size
        )
    else:
        upsample_factor = torch.tensor(
            upsample_factor, device=device, dtype=torch.float
        )
        shift = torch.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = torch.ceil(upsample_factor * 1.5)
        dftshift = torch.fix(upsampled_region_size / 2.0)
        sample_region_offset = dftshift - shift * upsample_factor
        cross_correlation = _upsampled_dft(
            image_product.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset,
            nb_spatial_dims,
        ).conj()
        cross_correlation = (cross_correlation * cross_correlation.conj()).real
        error, max_idx = torch.max(
            cross_correlation.reshape(*tuple(other_shapes), -1), dim=-1
        )
        maxima = unravel_index(
            max_idx,
            torch.as_tensor(
                cross_correlation.shape[-nb_spatial_dims:], device=max_idx.device
            ),
        )
        maxima -= dftshift

        shift += maxima / upsample_factor

        rg00 = torch.sum(
            (reference_image * reference_image.conj()),
            dim=tuple(
                range(reference_image.ndim - nb_spatial_dims, reference_image.ndim)
            ),
        )
        rf00 = torch.sum(
            (moving_image * moving_image.conj()),
            dim=tuple(range(moving_image.ndim - nb_spatial_dims, moving_image.ndim)),
        )

    error = torch.tensor([1.0], device=device) - error / (rg00.real * rf00.real)
    error = torch.sqrt(error.abs())

    return tuple([shift[..., i] for i in range(shift.size(-1))]), error, None


def phase_cross_correlation_broadcasted_skimage(
    reference_image: Array,
    moving_image: Array,
    *,
    upsample_factor: int = 1,
    space: str = "real",
    disambiguate: bool = False,
    reference_mask: Optional[Array] = None,
    moving_mask: Optional[Array] = None,
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
            return_error="always",
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


def phase_cross_correlation_broadcasted_cucim(
    reference_image: Array,
    moving_image: Array,
    *,
    upsample_factor: int = 1,
    space: str = "real",
    disambiguate: bool = False,
    reference_mask: Optional[Array] = None,
    moving_mask: Optional[Array] = None,
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


def phase_cross_correlation(
    reference_image: Array,
    moving_image: Array,
    *,
    upsample_factor: int = 1,
    space: str = "real",
    disambiguate: bool = False,
    reference_mask: Optional[Array] = None,
    moving_mask: Optional[Array] = None,
    overlap_ratio: float = 0.3,
    normalization: str = "phase",
    nb_spatial_dims: Optional[int] = None,
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

    Returns
    -------
    shift : array
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with
        the axis order of the input array.
    error : float
        Translation invariant normalized RMS error between
        ``reference_image`` and ``moving_image``. For masked cross-correlation
        this error is not available and NaN is returned if ``return_error``
        is "always".
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative). For masked cross-correlation
        this phase difference is not available and NaN is returned if
        ``return_error`` is "always".
    """
    xp = array_api_compat.array_namespace(reference_image, moving_image)
    if xp == array_api_compat.torch:
        func = phase_cross_correlation_broadcasted_pytorch
    elif xp == array_api_compat.numpy:
        func = phase_cross_correlation_broadcasted_skimage
    elif xp == array_api_compat.cupy:
        func = phase_cross_correlation_broadcasted_cucim

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
    )

    return shift, error, phasediff


def cartesian_prod(*arrays):
    xp = array_api_compat.array_namespace(*arrays)
    return xp.stack(xp.meshgrid(*arrays, indexing="ij"), axis=-1).reshape(
        -1, len(arrays)
    )


def discretize_sphere_uniformly(
    xp,
    N: int,
    M: int,
    symmetry: int = 1,
    product: bool = False,
    dtype=None,
    device=None,
) -> Tuple[Tuple[Array, Array, Array], Tuple[float, float]]:
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
    theta = xp.remainder(2 * xp.pi * i / goldenRatio, 2 * xp.pi)
    phi = xp.acos(1 - 2 * (i + epsilon) / N)
    psi = xp.linspace(0, 2 * np.pi / symmetry, M, device=device, dtype=dtype)
    if product:
        theta, psi2 = cartesian_prod(theta, psi).T
        phi, _ = cartesian_prod(phi, psi).T
        psi = psi2
    precision_axes = (
        (180 / xp.pi) * 2 * (xp.pi) ** 0.5 / N**0.5
    )  # aire autour d'un point = 4*pi/N
    precision_rot = (180 / xp.pi) * 2 * xp.pi / symmetry / M
    theta, phi, psi = theta * 180 / xp.pi, phi * 180 / xp.pi, psi * 180 / xp.pi
    return (theta, phi, psi), (precision_axes, precision_rot)


def normalize_patches(patches: torch.Tensor) -> torch.Tensor:
    """Normalize N patches by computing min/max for each patch
    Params: patches (torch.Tensor) of shape (N, ...)
    Returns: normalized_patches (torch.Tensor) of shape (N, ...)
    """
    N = patches.size(0)
    patch_shape = patches.shape[1:]
    flatten_patches = patches.view(N, -1)
    min_patch, _ = flatten_patches.min(dim=1)
    max_patch, _ = flatten_patches.max(dim=1)
    min_patch = min_patch.view(tuple([N] + [1] * len(patch_shape)))
    max_patch = max_patch.view(tuple([N] + [1] * len(patch_shape)))
    normalized_patches = (patches - min_patch) / (max_patch - min_patch)

    return normalized_patches


def disp3D(*ims, fig=None, axis_off=False):
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
            size=f"{100*ims[i].shape[0]/ims[i].shape[2]}%",
            pad="5%",
            sharex=axes[i],
        )
        ax_y = divider.append_axes(
            "bottom",
            size=f"{100*ims[i].shape[0]/ims[i].shape[1]}%",
            pad="5%",
            sharey=axes[i],
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

        axes[i].imshow(views[0], cmap="gray")
        ax_y.imshow(views[1], cmap="gray")
        ax_x.imshow(ndii.rotate(views[2], 90)[::-1], cmap="gray")

    if axis_off:
        for ax in axes:
            ax.set_axis_off()


def disp2D(fig, *ims, **imshowkwargs):
    h = int(np.floor(len(ims) ** 0.5))
    w = int(np.ceil(len(ims) / h))
    axes = fig.subplots(h, w)
    if type(axes) == np.ndarray:
        axes = axes.flatten()
        for ax in axes:
            ax.set_axis_off()
        for i in range(len(ims)):
            axes[i].imshow(ims[i], **imshowkwargs)
    else:
        axes.set_axis_off()
        axes.imshow(ims[0], **imshowkwargs)


def disp2D_compare(fig, *ims, **imshowkwargs):
    h = int(np.floor(len(ims) ** 0.5))
    w = int(np.ceil(len(ims) / h))
    axes = fig.subplots(h, w)
    if type(axes) == np.ndarray:
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


def create_psf(shape, cov, **kwargs):
    center = torch.floor(torch.as_tensor(shape, **kwargs) / 2)
    coords = torch.stack(
        torch.meshgrid(
            [torch.arange(0, shape[i], **kwargs) for i in range(len(shape))],
            indexing="ij",
        ),
        dim=-1,
    )
    psf = (
        torch.exp(
            -0.5
            * (coords[..., None, :] - center)
            @ torch.linalg.inv(cov)
            @ (coords[..., :, None] - center[:, None])
        )
        / torch.linalg.det(2 * torch.pi * cov) ** 0.5
    )
    return psf[..., 0, 0]


def are_volumes_aligned(vol1, vol2, atol=0.1):
    (dz, dy, dx), _, _ = phase_cross_correlation(
        vol1, vol2, upsample_factor=10, disambiguate=True, normalization=None
    )
    return dz <= atol and dy <= atol and dx <= atol
