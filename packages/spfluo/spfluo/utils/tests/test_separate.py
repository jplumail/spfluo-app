from functools import partial

import numpy as np
import pytest

from spfluo.data import generated_anisotropic
from spfluo.utils.separate import separate_centrioles, separate_centrioles_coords
from spfluo.utils.volume import assert_volumes_translated, resample


@pytest.fixture
def create_data():
    data = generated_anisotropic()
    im1, im2 = data["volumes"][[1, 2]]

    overlap = 15
    im = np.zeros((100 - overlap, 50, 50))
    im[:50] = im1
    im[50 - overlap :] += im2

    im = np.stack((im, np.random.randn(*im.shape)))
    im1 = np.stack((im1, im[1, :50]))
    im2 = np.stack((im2, im[1, 50 - overlap :]))
    good_channel = 0
    return im, (im1, im2), good_channel


@pytest.fixture
def create_data_monochannel(create_data):
    im, (im1, im2), _ = create_data
    return im[0], (im1[0], im2[0])


@pytest.fixture
def create_data_scaled(create_data_monochannel):
    im, (im1, im2) = create_data_monochannel
    scale = (3, 1, 1)
    im_resampled = resample(im, 1 / np.asarray(scale), order=3)
    im1_resampled = resample(im1, 1 / np.asarray(scale), order=3)
    im2_resampled = resample(im2, 1 / np.asarray(scale), order=3)
    return im_resampled, (im1_resampled, im2_resampled), scale


def common_im(im1, im2):
    common_shape = np.min((im1.shape, im2.shape), axis=0)
    return (
        im1[: common_shape[0], : common_shape[1], : common_shape[2]],
        im2[: common_shape[0], : common_shape[1], : common_shape[2]],
    )


def save_and_assert(save_result, im, im1, im2, im11, im22):
    save_result("original-im", im)
    save_result("im11", im11)
    save_result("im22", im22)

    ATOL = 1

    assert_volumes_translated_ = partial(assert_volumes_translated, atol=ATOL)
    same_order = assert_volumes_translated_(*common_im(im1, im11))
    if same_order:
        assert_volumes_translated_(*common_im(im2, im22))
    else:
        assert_volumes_translated_(*common_im(im1, im22))
        assert_volumes_translated_(*common_im(im2, im11))


def test_separate_simple(create_data_monochannel, save_result):
    im, (im1, im2) = create_data_monochannel
    im11, im22 = separate_centrioles(im, output_size=(50, 50, 50))

    save_and_assert(save_result, im, im1, im2, im11, im22)


def test_separate(create_data_monochannel, save_result):
    im, (im1, im2) = create_data_monochannel
    im11, im22 = separate_centrioles_coords(
        im, (50, 25, 25), (100, 50, 50), (50, 50, 50)
    )

    save_and_assert(save_result, im, im1, im2, im11, im22)


def test_separate_multichannel(create_data, save_result):
    im, (im1, im2), good_channel = create_data

    im11, im22 = separate_centrioles(im, (50, 50, 50), channel=good_channel)

    save_and_assert(
        save_result,
        im[good_channel],
        im1[good_channel],
        im2[good_channel],
        im11[good_channel],
        im22[good_channel],
    )


def test_separate_coords_multichannel(create_data, save_result):
    im, (im1, im2), good_channel = create_data

    im11, im22 = separate_centrioles_coords(
        im, (50, 25, 25), (100, 50, 50), (50, 50, 50), channel=good_channel
    )

    save_and_assert(
        save_result,
        im[good_channel],
        im1[good_channel],
        im2[good_channel],
        im11[good_channel],
        im22[good_channel],
    )


def test_separate_scaled(create_data_scaled, save_result):
    im, (im1, im2), scale = create_data_scaled

    im11, im22 = separate_centrioles_coords(
        im, (50, 25, 25), (100, 50, 50), (50, 50, 50), scale=scale
    )

    save_and_assert(save_result, im, im1, im2, im11, im22)
