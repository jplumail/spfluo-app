# Make the generated data available to all spfluo subpackages
from functools import partial
from typing import Any, Dict, Tuple

import torch
from spfluo.picking.modules.pretraining.generate.tests.test_data_generator import (
    generated_data,
    psf_array,
    groundtruth_array,
    particles,
    generated_particles_dir,
)

import pytest
import numpy as np


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
