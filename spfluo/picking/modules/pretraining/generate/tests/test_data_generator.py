from typing import Dict

import numpy as np
from scipy.ndimage import affine_transform, convolve

from spfluo.utils.transform import get_transform_matrix
from spfluo.utils.volume import are_volumes_aligned


def test_generation(psf_array, groundtruth_array, particles):
    D, H, W = groundtruth_array.shape
    dtype = groundtruth_array.dtype
    for k in particles:
        assert particles[k]["array"].shape == (D, H, W)
        assert particles[k]["array"].dtype == dtype
    assert psf_array.dtype == dtype


def test_poses(
    psf_array: np.ndarray,
    groundtruth_array: np.ndarray,
    particles: Dict[str, Dict[str, np.ndarray]],
):
    gt = groundtruth_array
    for k in particles:
        particle = particles[k]["array"]

        # H go from gt to particle (which is transformed)
        H = get_transform_matrix(
            particle.shape, particles[k]["rot"], particles[k]["trans"], degrees=True
        )

        # invert this because scipy's affine_transform works backward
        invH = np.linalg.inv(H)
        transformed_gt = affine_transform(gt, invH, order=1)

        # Apply the data model
        transformed_gt_blurred = convolve(
            transformed_gt, psf_array, mode="constant", cval=0.0
        )

        # Is the transformed groundtruth aligned with particle ?
        assert are_volumes_aligned(particle, transformed_gt_blurred)
