# Make the generated data available to all spfluo subpackages
import csv
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import tifffile
import torch

from spfluo.picking.modules.pretraining.generate.generate_data import generate_particles

D = 50
N = 10
anisotropy = (1.0, 1.0, 1.0)
DATA_DIR = Path(__file__).parent / "data"
pointcloud_path = DATA_DIR / "sample_centriole_point_cloud.csv"


@pytest.fixture(scope="session")
def generated_root_dir():
    root_dir = DATA_DIR / "generated"
    if not (root_dir / "particles").exists():
        root_dir.mkdir(exist_ok=True)
        np.random.seed(123)
        generate_particles(pointcloud_path, root_dir, D, N, anisotropy)
    return root_dir


@pytest.fixture(scope="session")
def psf_array(generated_root_dir):
    return tifffile.imread(generated_root_dir / "psf.tiff")


@pytest.fixture(scope="session")
def groundtruth_array(generated_root_dir):
    return tifffile.imread(generated_root_dir / "gt.tiff")


@pytest.fixture(scope="session")
def particles(generated_root_dir):
    content = csv.reader((generated_root_dir / "poses.csv").read_text().split("\n"))
    next(content)  # skip header
    data = {}
    for row in content:
        if len(row) == 7:
            data[row[0]] = {
                "array": tifffile.imread(generated_root_dir / row[0]),
                "rot": np.array(row[1:4], dtype=float),
                "trans": np.array(row[4:7], dtype=float),
            }
    return data


@pytest.fixture(scope="session")
def generated_data(
    psf_array, groundtruth_array, particles
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    return psf_array, groundtruth_array, particles


@pytest.fixture(scope="session")
def volumes_and_poses(
    generated_data: Tuple[Any, Any, Dict[str, Dict[str, np.ndarray]]]
) -> Tuple[np.ndarray, np.ndarray]:
    _, _, particles_dict = generated_data
    N = len(particles_dict)
    p0 = next(iter(particles_dict))
    D = particles_dict[p0]["array"].shape[0]
    volumes_arr = np.zeros((N, D, D, D))
    poses_arr = np.zeros((len(particles_dict), 6))
    for i, k in enumerate(particles_dict):
        p = particles_dict[k]
        rot = p["rot"]
        trans = p["trans"]
        poses_arr[i, :3] = rot
        poses_arr[i, 3:] = trans
        volumes_arr[i] = p["array"]
    return volumes_arr, poses_arr


@pytest.fixture(scope="session")
def generated_data_arrays(
    volumes_and_poses: Tuple[np.ndarray, np.ndarray],
    psf_array: np.ndarray,
    groundtruth_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    volumes_arr, poses_arr = volumes_and_poses
    return volumes_arr, poses_arr, psf_array, groundtruth_array


@pytest.fixture
def poses_with_noise(generated_data_arrays: Tuple[np.ndarray, ...]) -> np.ndarray:
    _, poses, _, _ = generated_data_arrays
    poses_noisy = poses.copy()
    sigma_rot, sigma_trans = 5, 0
    poses_noisy[:, :3] += np.random.randn(len(poses), 3) * sigma_rot
    poses_noisy[:, 3:] += np.random.randn(len(poses), 3) * sigma_trans
    return poses_noisy


gpu_as_tensor = partial(torch.as_tensor, device="cuda")


@pytest.fixture(scope="session")
def generated_data_pytorch(
    generated_data_arrays: Tuple[np.ndarray, ...]
) -> Tuple[torch.Tensor, ...]:
    return tuple(map(gpu_as_tensor, generated_data_arrays))


@pytest.fixture
def poses_with_noise_pytorch(poses_with_noise):
    return gpu_as_tensor(poses_with_noise)
