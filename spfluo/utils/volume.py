import itertools
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import scipy.ndimage as ndii
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import DTypeLike, NDArray
from scipy.ndimage import affine_transform as affine_transform_scipy
from scipy.ndimage import fourier_shift as fourier_shift_scipy
from skimage.registration import (
    phase_cross_correlation as phase_cross_correlation_skimage,
)

import spfluo
from spfluo.utils.array import Array, array_namespace, is_array_api_obj
from spfluo.utils.array import numpy as np
from spfluo.utils.transform import get_zoom_matrix

# Optional imports
if spfluo.has_cupy:
    from spfluo.utils.array import cupy

    from ._cupy_functions.volume import (
        affine_transform_batched_multichannel_cupy,
        fourier_shift_broadcasted_cupy,
        phase_cross_correlation_broadcasted_cucim,
    )

if spfluo.has_torch:
    from spfluo.utils.array import torch

    from ._torch_functions.volume import (
        affine_transform_batched_multichannel_pytorch,
        fourier_shift_broadcasted_pytorch,
        phase_cross_correlation_broadcasted_pytorch,
    )


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
    xp = array_namespace(input, matrix)
    has_output = False
    if is_array_api_obj(output):
        output = xp.asarray(output)
        has_output = True

    if batch is False:
        input = input[None]
        matrix = matrix[None]
        if has_output:
            output = output[None]
    if multichannel is False:
        input = input[:, None]

    if spfluo.has_torch and xp == torch:
        func = affine_transform_batched_multichannel_pytorch
    elif spfluo.has_cupy and xp == cupy:
        func = affine_transform_batched_multichannel_cupy
    elif xp == np:
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


def interpolate_to_size(
    volume: Array,
    output_size: Tuple[int, int, int],
    order=1,
    batch=False,
    multichannel=False,
) -> Array:
    """
    Used for padding. The zoom matrix will zoom-out from the image.
    """
    xp = array_namespace(volume)
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
    xp = array_namespace(input)
    if spfluo.has_torch and xp == torch:
        func = fourier_shift_broadcasted_pytorch
    elif xp == np:
        func = fourier_shift_broadcasted_scipy
    elif spfluo.has_cupy and xp == cupy:
        func = fourier_shift_broadcasted_cupy

    output = func(
        input,
        shift,
        n,
        axis,
        output,
    )
    return output


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
    xp = array_namespace(reference_image, moving_image)
    if spfluo.has_torch and xp == torch:
        func = phase_cross_correlation_broadcasted_pytorch
    elif xp == np:
        func = phase_cross_correlation_broadcasted_skimage
    elif spfluo.has_cupy and xp == cupy:
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
    xp = array_namespace(*arrays)
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


def are_volumes_aligned(vol1, vol2, atol=0.1):
    (dz, dy, dx), _, _ = phase_cross_correlation(
        vol1, vol2, upsample_factor=10, disambiguate=True, normalization=None
    )
    return dz <= atol and dy <= atol and dx <= atol
