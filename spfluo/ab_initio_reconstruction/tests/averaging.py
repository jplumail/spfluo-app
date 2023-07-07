import numpy as np

from spfluo.ab_initio_reconstruction.common_image_processing_methods.rotation_translation import (  # noqa: E501
    rotation,
)


def rotate_average(
    volumes: np.ndarray, transformation_matrices: np.ndarray
) -> np.ndarray:
    avrg = np.zeros_like(volumes[0])
    for v, t in zip(volumes, transformation_matrices):
        rot_mat = t[:3, :3]
        tvec = t[:3, 3]
        avrg += rotation(v, rot_mat, trans_vec=tvec)
    return avrg / len(volumes)
