# Make the generated data available to all spfluo subpackages
from typing import Tuple

import numpy as np
import pytest

from spfluo import data


@pytest.fixture(
    scope="session",
    params=[data.generated_isotropic(), data.generated_anisotropic()],
    ids=["isotropic", "anisotropic"],
)
def generated_data_all(
    request,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(request.param[k] for k in ["volumes", "poses", "psf", "gt"])


@pytest.fixture(scope="session")
def generated_data_anisotropic(
    request,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        data.generated_anisotropic()[k] for k in ["volumes", "poses", "psf", "gt"]
    )
