from typing import Tuple

import numpy as np
import scipy.ndimage as ndii
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.registration import phase_cross_correlation

from .memory import split_batch_func


def affine_transform(
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
        matrix (torch.Tensor): transform matrices of shape (N, 3), (N,3,3), (N,4,4) or (N,3,4).
        offset (float or torch.Tensor): offset of the grid.
        output_shape (tuple): shape of the output.
        output: not implemented
        order (int): must be 1. Only linear interpolation is implemented.
        mode (str): How to fill blank space. 'zeros', 'border' or 'reflection'
        cval (float): not implemented
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
            f"Matrix should be a tensor of shape {(N,3)}, {(N,3,3)}, {(N,4,4)} or {(N,3,4)}. Found matrix of shape {matrix.size()}"
        )

    if output_shape is None:
        output_shape = (D, H, W)

    if output is not None:
        raise NotImplementedError()

    if order > 1:
        raise NotImplementedError()

    pytorch_modes = ["zeros", "border", "reflection"]
    if mode not in pytorch_modes:
        raise NotImplementedError(f"Only {pytorch_modes} are available")

    if cval != 0:
        raise NotImplementedError()

    if not prefilter:
        raise NotImplementedError()

    return _affine_transform(input, rotMat, tvec, output_shape, mode, tensor_kwargs)


def _affine_transform(input, rotMat, tvec, output_shape, mode, tensor_kwargs):
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
    """Computes N dimensional inverse FFT of x in batch. Tries to avoid out-of-memory errors.

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


def fourier_shift(volume_freq: torch.Tensor, shift: torch.Tensor, nb_spatial_dims=None):
    """
    Args:
        volume (torch.Tensor): volume in the Fourier domain ({...}, [...])
            where [...] corresponds to the N spatial dimensions and {...} corresponds to the batched dimensions
        shift (torch.Tensor): shift to apply to the volume ({{...}}, N)
            where {{...}} corresponds to batched dimensions. {...} and {{...}} must be broadcastable.
        nb_spatial_dims (int): number of spatial dimensions N
    Returns:
        out (torch.Tensor): volume shifted in the Fourier domain
    """
    tensor_kwargs = {"device": volume_freq.device}
    if volume_freq.dtype == torch.complex128:
        tensor_kwargs["dtype"] = torch.float64
    elif volume_freq.dtype == torch.complex64:
        tensor_kwargs["dtype"] = torch.float32
    elif volume_freq.dtype == torch.complex32:
        tensor_kwargs["dtype"] = torch.float16
    else:
        print("Volume must be complex")
    if nb_spatial_dims is None:
        nb_spatial_dims = volume_freq.ndim
    spatial_shape = torch.as_tensor(
        volume_freq.size()[-nb_spatial_dims:], **tensor_kwargs
    )
    assert shift.size(-1) == nb_spatial_dims
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
    out = volume_freq * torch.exp(-1j * 2 * torch.pi * phase_shift)

    return out


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
        x (torch.Tensor) of shape (B, ...) where (...) corresponds to the N spatial dimensions
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


def dftregistrationND(
    reference: torch.Tensor,
    moving_images: torch.Tensor,
    nb_spatial_dims: int = None,
    upsample_factor: int = 1,
    normalization: str = "phase",
) -> torch.Tensor:
    """Phase cross-correlation between a reference and moving_images
    Params:
        reference (torch.Tensor): image of shape ({...}, [...]) where [...] corresponds to the N spatial dimensions
        moving_images (torch.Tensor): images to register of shape ({{...}}, [...]). {...} and {{...}} are broadcasted to (...)
        nb_spatial_dims (int): specify the N spatial dimensions
        upsample_factor (float): upsampling factor. Images will be registered up to 1/upsample_factor.
    Returns:
        error (torch.Tensor): tensor of shape (...)
        shift (Tuple[torch.Tensor]): tuple of N tensors of size (...)
    """
    device = reference.device
    output_shape = torch.as_tensor(
        torch.broadcast_shapes(reference.size(), moving_images.size())
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
        reference, moving_images, normalization, nb_spatial_dims=nb_spatial_dims
    )

    # Now change shifts so that they represent relative shifts and not indices
    spatial_shapes_broadcasted = torch.broadcast_to(spatial_shapes, shift.size()).to(
        device
    )
    shift[shift > midpoints] -= spatial_shapes_broadcasted[shift > midpoints]

    spatial_size = torch.prod(spatial_shapes).type(reference.dtype)

    if upsample_factor == 1:
        rg00 = (
            torch.sum(
                (reference * reference.conj()),
                dim=tuple(range(reference.ndim - nb_spatial_dims, reference.ndim)),
            )
            / spatial_size
        )
        rf00 = (
            torch.sum(
                (moving_images * moving_images.conj()),
                dim=tuple(
                    range(moving_images.ndim - nb_spatial_dims, moving_images.ndim)
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
            (reference * reference.conj()),
            dim=tuple(range(reference.ndim - nb_spatial_dims, reference.ndim)),
        )
        rf00 = torch.sum(
            (moving_images * moving_images.conj()),
            dim=tuple(range(moving_images.ndim - nb_spatial_dims, moving_images.ndim)),
        )

    error = torch.tensor([1.0], device=device) - error / (rg00.real * rf00.real)
    error = torch.sqrt(error.abs())

    return error, tuple([shift[..., i] for i in range(shift.size(-1))])


def discretize_sphere_uniformly(
    N: int, M: int, symmetry: int = 1, product: bool = False, **tensor_kwargs
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], float]:
    """Generates a list of the two first euler angles that describe a uniform discretization of the sphere
    with the Fibonnaci sphere algorithm.
    Params:
        N, the number of axes (two first euler angles)
        M, the number of rotations around the axes (third euler angle)
        symmetry, the order of symmetry to reduce the range of the 3rd angle. Default to 1, no symmetry
        product, if True return the cartesian product between the axes and the rotations
    Returns: (theta, phi, psi), precision
        precision, a float representing an approximation of the sampling done
        (theta, phi, psi), a tuple of 1D tensors containing the 1st, 2nd and 3rd euler angles
            theta.shape == phi.shape == (N,)
            psi.shape == (M,)
        if product is false,
            theta.shape == phi.shape == psi.shape == (N*M,)
    """
    epsilon = 0.5
    goldenRatio = (1 + 5**0.5) / 2
    i = torch.arange(0, N, **tensor_kwargs)
    theta = torch.remainder(2 * torch.pi * i / goldenRatio, 2 * torch.pi)
    phi = torch.acos(1 - 2 * (i + epsilon) / N)
    psi = torch.linspace(0, 2 * np.pi / symmetry, M, **tensor_kwargs)
    if product:
        theta, psi2 = torch.cartesian_prod(theta, psi).T
        phi, _ = torch.cartesian_prod(phi, psi).T
        psi = psi2
    precision = 360 / (torch.pi * N) ** 0.5
    theta, phi, psi = theta * 180 / torch.pi, phi * 180 / torch.pi, psi * 180 / torch.pi
    return (theta, phi, psi), precision


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


def disp3D(fig, *ims, axis_off=False):
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
