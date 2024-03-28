import numpy as np
import pytest
from scipy.ndimage import affine_transform, convolve

from spfluo import data
from spfluo.utils.transform import get_transform_matrix
from spfluo.utils.volume import are_volumes_aligned


def test_generation():
    d = data.generated_anisotropic()
    D, H, W = d["gt"].shape
    dtype = d["gt"].dtype
    N, x = d["poses"].shape
    assert x == 6
    assert d["volumes"].shape == (N, D, H, W)
    assert d["volumes"].dtype == dtype
    assert d["psf"].dtype == dtype


@pytest.mark.slow
@pytest.mark.parametrize(
    "generated_data", [data.generated_anisotropic(), data.generated_isotropic()]
)
def test_poses(generated_data):
    gt = generated_data["gt"]
    for particle, pose in zip(generated_data["volumes"], generated_data["poses"]):
        # H go from gt to particle (which is transformed)
        H = get_transform_matrix(particle.shape, pose[:3], pose[3:], degrees=True)

        # invert this because scipy's affine_transform works backward
        invH = np.linalg.inv(H)
        transformed_gt = affine_transform(gt, invH, order=1)

        # Apply the data model
        transformed_gt_blurred = convolve(
            transformed_gt, generated_data["psf"], mode="constant", cval=0.0
        )

        # Is the transformed groundtruth aligned with particle ?
        assert are_volumes_aligned(particle, transformed_gt_blurred)
