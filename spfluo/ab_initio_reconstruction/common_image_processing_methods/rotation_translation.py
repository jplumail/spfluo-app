import array_api_compat
import numpy as np

from spfluo.utils._array import Array
from spfluo.utils.volume import affine_transform as affine_transform_spfluo

interp_order = 3


def rotation(volumes: Array, transform: Array, order: int = 1, inverse: bool = False):
    """
    transform: homogeneous transform
    """
    xp = array_api_compat.array_namespace(volumes, transform)
    if not inverse:  # scipy's affine_transform do inverse transform by default
        transform = xp.linalg.inv(transform)
    return affine_transform_spfluo(
        volumes, transform, order=order, prefilter=False, batch=True
    )


def conversion_2_first_eulers_angles_cartesian(theta, phi, degrees=True):
    if degrees:
        theta = theta * np.pi / 180
        phi = phi * np.pi / 180
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return x, y, z
