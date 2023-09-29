from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from spfluo.utils.memory import split_batch_func


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

    if order > 1 and prefilter:
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
    reference_image: torch.Tensor,
    moving_image: torch.Tensor,
    *,
    upsample_factor: int = 1,
    space: str = "real",
    disambiguate: bool = False,
    reference_mask: Optional[torch.Tensor] = None,
    moving_mask: Optional[torch.Tensor] = None,
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
