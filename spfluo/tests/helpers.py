import numpy

import spfluo
from spfluo.utils.array import Array, to_numpy
from spfluo.utils.array import numpy as np
from spfluo.utils.volume import phase_cross_correlation

testing_libs = [(np, None)]
ids = ["numpy"]

if spfluo.has_cupy:
    from spfluo.utils.array import cupy

    testing_libs.append((cupy, None))
    ids.append("cupy")
if spfluo.has_torch:
    from spfluo.utils.array import torch

    testing_libs.append((torch, "cpu"))
    ids.append("torch-cpu")
    if spfluo.has_torch_cuda:
        testing_libs.append((torch, "cuda"))
        ids.append("torch-cuda")


def assert_volumes_aligned(
    vol1: Array, vol2: Array, atol: float = 0.1, nb_spatial_dims: int = 3
):
    (dz, dy, dx), _, _ = phase_cross_correlation(
        vol1,
        vol2,
        upsample_factor=10,
        disambiguate=False,
        normalization=None,
        nb_spatial_dims=nb_spatial_dims,
    )
    n = (dz**2 + dy**2 + dx**2) ** 0.5
    assert (n <= atol).all(), f"{n.max()} > {atol}"


def assert_allclose(actual: Array, desired: Array, rtol=1e-7, atol=0):
    numpy.testing.assert_allclose(
        to_numpy(actual), to_numpy(desired), rtol=rtol, atol=atol
    )
