from functools import partial

import array_api_compat.cupy
import array_api_compat.numpy
import array_api_compat.torch
import numpy as np
import pytest
import torch
from array_api_compat import array_namespace
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.ndimage import affine_transform
from scipy.ndimage import fourier_shift as fourier_shift_scipy
from scipy.spatial.transform import Rotation as R
from skimage import data, util
from skimage.registration import (
    phase_cross_correlation as phase_cross_correlation_skimage,
)

import spfluo.utils
from spfluo.utils.volume import fourier_shift, phase_cross_correlation


def assert_allclose(a, b, rtol=1e-7, atol=0):
    xp = array_namespace(a, b)
    assert xp.all(xp.abs(a - b) <= atol + rtol * xp.abs(b))


##################################################################
# Test phase_cross_correlation against the scikit-image function #
##################################################################


phase_cross_correlation_skimage = partial(
    phase_cross_correlation_skimage, return_error="always"
)


@settings(deadline=None)
@given(
    translation=arrays(
        float,
        (3,),
        elements=st.floats(
            min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
        ),
    ),
    upsample_factor=st.integers(min_value=1, max_value=100),
    normalization=st.sampled_from(["phase", None]),
)
@pytest.mark.parametrize(
    "xp, device",
    [
        (array_api_compat.numpy, None),
        (array_api_compat.cupy, None),
        (array_api_compat.torch, "cpu"),
        (array_api_compat.torch, "cuda"),
    ],
)
@pytest.mark.parametrize("image", [data.camera(), data.cells3d()[:, 0, :60, :60]])
def test_correctness_phase_cross_correlation(
    xp, device, image, translation, upsample_factor, normalization
):
    reference_image = np.fft.fftn(util.img_as_float(image))
    translation = translation[: reference_image.ndim]
    moving_image = fourier_shift(reference_image, translation)

    reference_image_xp = xp.asarray(reference_image, device=device)
    moving_image_xp = xp.asarray(moving_image, device=device)

    (shift, error, phasediff), (shift_skimage, error_skimage, phasediff_skimage) = [
        func(
            reference_image_,
            moving_image_,
            space="fourier",
            upsample_factor=upsample_factor,
            normalization=normalization,
        )
        for func, reference_image_, moving_image_ in [
            (phase_cross_correlation, reference_image_xp, moving_image_xp),
            (phase_cross_correlation_skimage, reference_image, moving_image),
        ]
    ]

    for i in range(reference_image.ndim):
        assert_allclose(
            xp.asarray(shift[i]), xp.asarray(shift_skimage[i]), atol=1 / upsample_factor
        )
    assert_allclose(error, xp.asarray(error_skimage), atol=1e7, rtol=0.01)
    if xp != array_api_compat.torch:
        assert_allclose(phasediff, xp.asarray(phasediff_skimage), atol=1e7, rtol=0.01)


def test_broadcasting_phase_cross_correlation():  # TODO
    pass


####################################################
# Test affine_transform against the scipy function #
####################################################


def affine_transform_gpu(
    vol,
    mat,
    offset=0.0,
    output_shape=None,
    device="cpu",
    batch=False,
    multichannel=False,
):
    out = spfluo.utils.volume.affine_transform(
        torch.as_tensor(vol, device=device),
        torch.as_tensor(mat, device=device),
        offset=offset,
        output_shape=output_shape,
        batch=batch,
        multichannel=multichannel,
        prefilter=False,
        order=1,
    )
    return out.cpu().numpy()


def create_2d_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def is_affine_close(im1, im2):
    """Scipy's and pytorch interpolations at borders don't behave equivalently
    So we add a margin"""
    D, H, W = im1.shape
    return np.isclose(im1, im2).sum() > (D * W * H - 2 * (H * D + D * W + W * H))


def test_affine_transform_simple():
    N = 10
    image = util.img_as_float(data.cells3d()[:, 1, :, :])
    matrices = np.empty((N, 4, 4))
    for i in range(N):
        matrices[i] = np.eye(4)
        matrices[i, :3, :3] = np.eye(3) + R.random().as_matrix() * 0.1
        matrices[i, :3, 3] = np.random.randn(3)

    out_scipy = [affine_transform(image, m, order=1) for m in matrices]
    out_pt = list(affine_transform_gpu(np.stack([image] * N), matrices, batch=True))
    assert all([is_affine_close(x, y) for x, y in zip(out_scipy, out_pt)])


def test_affine_transform_output_shape():
    output_shapes = [
        (64, 32, 32),
        (32, 32, 32),
        (64, 128, 256),
        (128, 128, 57),
        (57, 56, 55),
    ]
    image = util.img_as_float(data.cells3d()[:, 1, :, :])
    matrix = np.eye(4)
    matrix[:3, :3] = np.eye(3) + R.random().as_matrix() * 0.1
    matrix[:3, 3] = np.random.randn(3)

    out_scipy = [
        affine_transform(image, matrix, order=1, output_shape=o) for o in output_shapes
    ]
    out_pt = [
        affine_transform_gpu(image[None], matrix[None], output_shape=o, batch=True)
        for o in output_shapes
    ]
    assert all([is_affine_close(x, y) for x, y in zip(out_scipy, out_pt)])


def test_affine_transform_offset():
    N = 10
    image = util.img_as_float(data.cells3d()[:, 1, :, :])
    matrix = np.eye(3) + R.random().as_matrix() * 0.1
    matrices = np.stack([matrix] * N)
    offsets = np.random.randn(N, 3)

    out_scipy = [affine_transform(image, matrix, order=1, offset=o) for o in offsets]
    out_pt = affine_transform_gpu(
        np.stack([image] * N), matrices, offset=offsets, batch=True
    )
    assert all([is_affine_close(x, y) for x, y in zip(out_scipy, out_pt)])


#################################################
# Test fourier_shift against the scipy function #
#################################################


@settings(deadline=None)
@given(
    shift=arrays(
        float,
        (3,),
        elements=st.floats(
            min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
        ),
    ),
)
@pytest.mark.parametrize(
    "xp, device",
    [
        (array_api_compat.numpy, None),
        (array_api_compat.cupy, None),
        (array_api_compat.torch, "cpu"),
        (array_api_compat.torch, "cuda"),
    ],
)
@pytest.mark.parametrize("image", [data.camera(), data.cells3d()[:, 0, :60, :60]])
def test_correctness_fourier_shift(
    xp,
    device,
    image,
    shift,
):
    input = np.fft.fftn(util.img_as_float(image))
    shift = shift[: image.ndim]

    input_xp = xp.asarray(input, device=device)
    shift_xp = xp.asarray(shift, device=device)

    output = fourier_shift(input_xp, shift_xp)
    output_scipy = fourier_shift_scipy(input, shift)

    assert_allclose(
        output,
        xp.asarray(output_scipy, device=device),
    )


def test_broadcasting_fourier_shift():  # TODO
    pass
