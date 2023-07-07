# Make the generated data available to all spfluo subpackages
import csv
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import tifffile
import torch

from spfluo.picking.modules.pretraining.generate.data_generator import (
    DataGenerationConfig,
    DataGenerator,
)

D = 50
N = 10
anisotropy = (1.0, 1.0, 1.0)
data_dir = Path(__file__).parent / "data"
pointcloud_path = data_dir / "sample_centriole_point_cloud.csv"


@pytest.fixture(scope="session")
def generated_particles_dir():
    particles_dir = data_dir / "generated_particles"
    if not particles_dir.exists():
        np.random.seed(123)
        config = DataGenerationConfig()
        config.augmentation.max_translation = 0
        config.io.point_cloud_path = pointcloud_path
        config.io.extension = "tiff"
        config.voxelisation.image_shape = D
        config.voxelisation.max_particle_dim = int(0.6 * D)
        config.voxelisation.num_particles = N
        config.voxelisation.bandwidth = 17
        config.sensor.anisotropic_blur_sigma = anisotropy
        config.augmentation.rotation_proba = 1
        config.augmentation.shrink_range = (1.0, 1.0)
        gen = DataGenerator(config)
        particles_dir.mkdir()
        gt_path = particles_dir / "gt.tiff"
        gen.save_psf(particles_dir / "psf.tiff")
        gen.save_groundtruth(gt_path)
        gen.create_particles(particles_dir, output_extension="tiff")

    return particles_dir


@pytest.fixture(scope="session")
def psf_array(generated_particles_dir):
    return tifffile.imread(generated_particles_dir / "psf.tiff")


@pytest.fixture(scope="session")
def groundtruth_array(generated_particles_dir):
    return tifffile.imread(generated_particles_dir / "gt.tiff")


@pytest.fixture(scope="session")
def particles(generated_particles_dir):
    content = csv.reader(
        (generated_particles_dir / "poses.csv").read_text().split("\n")
    )
    next(content)  # skip header
    data = {}
    for row in content:
        if len(row) == 7:
            data[row[0]] = {
                "array": tifffile.imread(generated_particles_dir / row[0]),
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
