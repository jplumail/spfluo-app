# Make the generated data available to all spfluo subpackages
import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pytest
import tifffile

from spfluo import data


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--image-directory", action="store", default=None)


@pytest.fixture(scope="session")
def image_directory(pytestconfig: pytest.Config):
    dir = pytestconfig.getoption("--image-directory")
    if dir:
        dir = Path(dir)
        if not dir.exists():
            dir.mkdir()
    else:
        dir = None
    return dir


@pytest.fixture(scope="function")
def save_result(image_directory: Union[Path, None], request: pytest.FixtureRequest):
    base_name = os.path.split(request.node.nodeid)[1]

    def inner(name: str, arr: np.ndarray):
        saved = False
        if image_directory:
            tifffile.imwrite(
                str(image_directory / (base_name + "-" + name + ".tiff")), arr
            )
            saved = True
        return saved

    return inner


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
def generated_data_anisotropic() -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    d = data.generated_anisotropic()
    return tuple(d[k] for k in ["volumes", "poses", "psf", "gt"])
