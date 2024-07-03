from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


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
            Should be of floating point dtype
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
    dtype = input.dtype
    (device,) = set((input.device, matrix.device))
    tensor_kwargs = dict(device=device, dtype=dtype)

    if isinstance(offset, float):
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
        raise NotImplementedError("order > 1 is not implemented")

    pytorch_modes = {"constant": "zeros", "nearest": "border", "reflect": "reflection"}
    if mode not in pytorch_modes:
        raise NotImplementedError(f"Only {pytorch_modes.keys()} are available")
    pt_mode = pytorch_modes[mode]

    if cval != 0:
        raise NotImplementedError()

    if order > 1 and prefilter:
        raise NotImplementedError()

    return _affine_transform(
        input, rotMat.type(input.dtype), tvec, output_shape, pt_mode, **tensor_kwargs
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
    grid = (
        (rotMat[:, None, None, None] @ ((grid - c)[None, ..., None]))[..., 0]
        + c
        + tvec[:, None, None, None, :]
    )
    grid = -1 + 1 / input_shape + 2 * grid / input_shape
    rotated_vol = F.grid_sample(
        input,
        grid[:, :, :, :, [2, 1, 0]],
        mode="bilinear",
        align_corners=False,
        padding_mode=mode,
    )
    return rotated_vol


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
