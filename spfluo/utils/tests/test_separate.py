from functools import partial

import numpy as np

import pytest
from spfluo.data import generated_anisotropic
from spfluo.utils.separate import separate_centrioles
from spfluo.utils.volume import are_volumes_translated


@pytest.fixture
def create_data():
    data = generated_anisotropic()
    im1, im2 = data["volumes"][:2]

    im = np.zeros((100, 50, 50))
    im[:50] = im1
    im[50:] = im2
    return im, (im1, im2)


def common_im(im1, im2):
    common_shape1 = np.min((im1.shape, im2.shape), axis=0)
    return (
        im1[: common_shape1[0], : common_shape1[1], : common_shape1[2]],
        im2[: common_shape1[0], : common_shape1[1], : common_shape1[2]],
    )


def test_separate(create_data, save_result):
    im, (im1, im2) = create_data
    im11, im22 = separate_centrioles(im)

    save_result("im1", im1)
    save_result("im11", im11)
    save_result("im2", im2)
    save_result("im22", im22)

    are_volumes_translated_ = partial(are_volumes_translated, atol=0.2)
    same_order = are_volumes_translated_(*common_im(im1, im11))
    if same_order:
        assert are_volumes_translated_(*common_im(im2, im22))
    else:
        assert are_volumes_translated_(*common_im(im1, im22))
        assert are_volumes_translated_(*common_im(im2, im11))
