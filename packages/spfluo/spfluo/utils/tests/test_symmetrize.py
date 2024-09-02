import numpy as np
import pytest
import torch

from spfluo.utils.symmetrize_particle import symmetrize
from spfluo.utils.transform import get_transform_matrix
from spfluo.utils.volume import affine_transform, are_volumes_aligned


@pytest.mark.skip(reason="to fix")
def test_symmetrize_simple(generated_data_anisotropic):
    volumes, _, psf, groundtruth = generated_data_anisotropic
    vol = volumes[6]
    # translate vol
    vol = affine_transform(
        vol,
        np.linalg.inv(
            get_transform_matrix(
                vol.shape,
                np.array([0.0, 0.0, 0.0]),
                np.array([-7.0, 2.0, 5.0]),
                degrees=True,
            )
        ),
    )
    recon = symmetrize(
        torch.as_tensor(vol), (2.0, 5.0), 9, torch.as_tensor(psf), torch.as_tensor(1e-3)
    ).numpy()

    assert are_volumes_aligned(groundtruth, recon, atol=4)
