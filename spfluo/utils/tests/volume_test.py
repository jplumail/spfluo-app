from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.ndimage import affine_transform as affine_transform_scipy
from scipy.ndimage import fourier_shift as fourier_shift_scipy
from scipy.ndimage import shift as shift_scipy
from skimage import data, util
from skimage.registration import (
    phase_cross_correlation as phase_cross_correlation_skimage,
)

from spfluo.tests.helpers import assert_allclose, random_pose, testing_libs
from spfluo.utils.array import numpy as np
from spfluo.utils.array import to_numpy
from spfluo.utils.transform import get_transform_matrix_from_pose
from spfluo.utils.volume import affine_transform, fourier_shift, phase_cross_correlation

if TYPE_CHECKING:
    from spfluo.utils.array import array_api_module


##################################################################
# Test phase_cross_correlation against the scikit-image function #
##################################################################


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
    space=st.sampled_from(["fourier", "real"]),
    normalization=st.sampled_from(["phase", None]),
    half_precision=st.sampled_from([True, False]),
)
@pytest.mark.parametrize(
    "xp, device",
    testing_libs,
)
@pytest.mark.parametrize("image", [data.camera(), data.cells3d()[:, 0, :60, :60]])
def test_correctness_phase_cross_correlation(
    xp: "array_api_module",
    device,
    image,
    translation,
    upsample_factor,
    space,
    normalization,
    half_precision,
):
    translation = translation[: image.ndim]
    if space == "fourier":
        reference_image = np.fft.fftn(util.img_as_float(image))
        moving_image = fourier_shift(reference_image, translation)
        if half_precision:
            reference_image = reference_image.astype(np.complex64)
            moving_image = moving_image.astype(np.complex64)
    else:
        reference_image = util.img_as_float(image)
        moving_image = shift_scipy(reference_image, translation)
        if half_precision:
            reference_image = reference_image.astype(np.float32)
            moving_image = moving_image.astype(np.float32)

    reference_image_xp = xp.asarray(reference_image, device=device)
    moving_image_xp = xp.asarray(moving_image, device=device)

    (shift, error, phasediff), (shift_skimage, error_skimage, phasediff_skimage) = [
        func(
            reference_image_,
            moving_image_,
            space=space,
            upsample_factor=upsample_factor,
            normalization=normalization,
        )
        for func, reference_image_, moving_image_ in [
            (phase_cross_correlation, reference_image_xp, moving_image_xp),
            (phase_cross_correlation_skimage, reference_image, moving_image),
        ]
    ]
    translation = to_numpy(translation)
    shift_skimage = -np.asarray(to_numpy(*shift_skimage))
    shift = -np.asarray(to_numpy(*shift))
    norm_error_vec_skimage = np.linalg.vector_norm(translation - shift_skimage)
    norm_error_vec = np.linalg.vector_norm(translation - shift)
    if norm_error_vec_skimage < norm_error_vec:
        assert_allclose(
            norm_error_vec, norm_error_vec_skimage, rtol=0.1, atol=1 / upsample_factor
        )  # atol=0.1
    assert_allclose(error, xp.asarray(error_skimage), rtol=1e-3, atol=1e-2)
    # Phasediff not tested because not implemented in pytorch


def test_broadcasting_phase_cross_correlation():  # TODO
    pass


@pytest.fixture
def cells3d_translated():
    image = np.transpose(
        data.cells3d()[:, :, :60, :60], (1, 0, 2, 3)
    )  # shape (C, D, H, W)
    translation = (5, -4, 10)

    reference_image = np.fft.fftn(util.img_as_float(image), axes=(1, 2, 3))
    moving_image = fourier_shift(reference_image, np.asarray(translation))
    return reference_image, moving_image, translation


@pytest.mark.parametrize("xp, device", testing_libs)
def test_multichannel_phase_cross_correlation_1(
    xp: "array_api_module", device, cells3d_translated
):
    reference_image, moving_image, translation = cells3d_translated
    reference_image_xp = xp.asarray(reference_image, device=device)
    moving_image_xp = xp.asarray(moving_image, device=device)

    shift, error, phasediff = phase_cross_correlation(
        reference_image_xp,
        moving_image_xp,
        space="fourier",
        nb_spatial_dims=3,
        multichannel=True,
    )
    assert len(shift) == 3
    for i in range(len(shift)):
        assert shift[i].ndim == 0
    assert error.ndim == 0
    assert_allclose(xp.stack(shift), -xp.asarray(translation))
    assert_allclose(error, xp.asarray(1.0), rtol=1e-5)


# With zeros on first channel
@pytest.mark.parametrize("xp, device", testing_libs)
def test_multichannel_phase_cross_correlation_2(
    xp: "array_api_module", device, cells3d_translated
):
    reference_image, moving_image, translation = cells3d_translated
    reference_image_xp = xp.asarray(reference_image, device=device)
    moving_image_xp = xp.asarray(moving_image, device=device)
    moving_image_xp_ = xp.asarray(moving_image_xp, copy=True)
    moving_image_xp_[0, ...] = 0.0
    shift, error, phasediff = phase_cross_correlation(
        reference_image_xp,
        moving_image_xp_,
        space="fourier",
        nb_spatial_dims=3,
        multichannel=True,
    )
    assert len(shift) == 3
    for i in range(len(shift)):
        assert shift[i].ndim == 0
    assert error.ndim == 0
    assert_allclose(xp.stack(shift), -xp.asarray(translation))
    assert_allclose(error, xp.asarray(1.0), rtol=1e-5)


# With upsampling
@pytest.mark.parametrize("xp, device", testing_libs)
def test_multichannel_phase_cross_correlation_upsampling(
    xp: "array_api_module", device
):
    image = np.transpose(
        data.cells3d()[:, :, :60, :60], (1, 0, 2, 3)
    )  # shape (C, D, H, W)
    translation = (5.3, -4.1, 10.4)

    reference_image = np.fft.fftn(util.img_as_float(image), axes=(1, 2, 3))
    moving_image = fourier_shift(reference_image, np.asarray(translation))

    reference_image_xp = xp.asarray(reference_image, device=device)
    moving_image_xp = xp.asarray(moving_image, device=device)
    shift, error, phasediff = phase_cross_correlation(
        reference_image_xp,
        moving_image_xp,
        space="fourier",
        nb_spatial_dims=3,
        multichannel=True,
        upsample_factor=10,
    )
    assert len(shift) == 3
    for i in range(len(shift)):
        assert shift[i].ndim == 0
    assert error.ndim == 0
    assert_allclose(xp.stack(shift), -xp.asarray(translation))
    assert_allclose(error, xp.asarray(1.0), rtol=1e-5)


####################################################
# Test affine_transform against the scipy function #
####################################################


def is_affine_close(im1, im2):
    """Scipy's and pytorch interpolations at borders don't behave equivalently
    So we add a margin"""
    D, H, W = im1.shape
    return np.isclose(im1, im2).sum() > (D * W * H - 2 * (H * D + D * W + W * H))


@settings(deadline=None)  # necessary for GPU testing
@given(
    pose=random_pose(),
    order=st.just(1),
    mode=st.sampled_from(["constant", "nearest", "reflect"]),
    cval=st.just(0.0),
    prefilter=st.just(True),
    d=st.data(),
)
@pytest.mark.parametrize(
    "xp, device",
    testing_libs,
)
@pytest.mark.parametrize("input", [util.img_as_float(data.cells3d()[:, 0, :60, :60])])
def test_correctness_affine_transform(
    xp: "array_api_module", input, device, pose, order, mode, cval, prefilter, d
):
    output_shape = d.draw(
        st.tuples(
            *[
                st.integers(min_value=input.shape[i] // 2, max_value=input.shape[i] * 2)
                for i in range(input.ndim)
            ],
        )
    )
    matrix = get_transform_matrix_from_pose(output_shape, pose)
    input_xp = xp.asarray(input, device=device)
    matrix_xp = xp.asarray(matrix, device=device)

    output_xp, output = [
        func(
            input_,
            matrix_,
            output_shape=output_shape,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
        )
        for func, input_, matrix_ in [
            (affine_transform, input_xp, matrix_xp),
            (affine_transform_scipy, input, matrix),
        ]
    ]

    assert is_affine_close(to_numpy(output_xp), output)


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
    testing_libs,
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
        rtol=1e-5,
    )


def test_broadcasting_fourier_shift():  # TODO
    pass
