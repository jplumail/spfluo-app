from typing import TYPE_CHECKING

import numpy
from hypothesis import assume
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation as R

import spfluo
from spfluo.utils.array import array_namespace, get_cupy, get_torch, to_numpy
from spfluo.utils.array import numpy as np
from spfluo.utils.volume import phase_cross_correlation

if TYPE_CHECKING:
    from spfluo.utils.array import Array


testing_libs = [(np, "cpu")]
ids = ["numpy"]

try:
    import array_api_strict as xp
    from array_api_strict._array_object import CPU_DEVICE

    testing_libs.append((xp, CPU_DEVICE))
    ids.append("array-api-strict")
except ModuleNotFoundError:
    pass


if spfluo.has_cupy():
    cupy = get_cupy()
    testing_libs.append((cupy, cupy.cuda.Device(0)))
    ids.append("cupy")

if spfluo.has_torch():
    torch = get_torch()
    testing_libs.append((torch, "cpu"))
    ids.append("torch-cpu")
    if spfluo.has_torch_cuda():
        testing_libs.append((torch, "cuda"))
        ids.append("torch-cuda")


@st.composite
def random_pose(
    draw,
    quaternions=st.tuples(
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
    ),
    translation=st.tuples(
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
        st.floats(0, 1, allow_infinity=False, allow_nan=False),
    ),
    translation_magnitude=st.floats(0, 1, allow_infinity=False, allow_nan=False),
):
    quaternions = draw(quaternions)
    translation = draw(translation)
    translation_magnitude = draw(translation_magnitude)

    assume(np.linalg.norm(quaternions) > 0)
    assume(not np.isinf(np.linalg.norm(quaternions)))
    assume(np.linalg.norm(translation) > 0)
    assume(not np.isinf(np.linalg.norm(translation)))

    euler = R.from_quat(quaternions).as_euler("XZX", degrees=True)
    translation /= np.linalg.norm(translation)
    translation *= translation_magnitude
    pose = np.concatenate((euler, translation))
    return pose


def assert_volumes_aligned(
    vol1: "Array",
    vol2: "Array",
    atol: float = 0.1,
    nb_spatial_dims: int = 3,
    multichannel: bool = False,
):
    xp = array_namespace(vol1, vol2)
    (dz, dy, dx), _, _ = phase_cross_correlation(
        xp.astype(vol1, vol2.dtype),
        vol2,
        upsample_factor=10,
        disambiguate=False,
        normalization=None,
        nb_spatial_dims=nb_spatial_dims,
        multichannel=multichannel,
    )
    n = (dz**2 + dy**2 + dx**2) ** 0.5
    assert xp.all(n <= atol), f"{xp.max(n)} > {atol}"


def assert_allclose(actual: "Array", desired: "Array", rtol=1e-7, atol=0):
    numpy.testing.assert_allclose(
        to_numpy(actual), to_numpy(desired), rtol=rtol, atol=atol
    )
