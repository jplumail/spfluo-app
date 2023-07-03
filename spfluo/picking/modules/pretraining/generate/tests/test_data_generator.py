from typing import Any, Dict, Tuple
import numpy as np
from spfluo.picking.modules.pretraining.generate.data_generator import DataGenerationConfig, DataGenerator
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.spatial.transform import Rotation
from skimage.registration import phase_cross_correlation
import pytest
import tifffile
import csv


D = 50
N = 20

@pytest.fixture(scope="session")
def generated_particles(tmp_path_factory):
    config = DataGenerationConfig()
    config.augmentation.max_translation = 0.1
    config.io.point_cloud_path = "/home/plumail/Téléchargements/sample_centriole_point_cloud.csv"
    config.io.extension = "tiff"
    config.voxelisation.image_shape = D
    config.voxelisation.max_particle_dim = int(0.6*D)
    config.voxelisation.num_particles = N
    config.voxelisation.bandwidth = 17
    config.sensor.anisotropic_blur_sigma = (5, 1, 1)
    config.augmentation.rotation_proba = 1
    config.augmentation.max_translation = 100
    config.augmentation.shrink_range = (1., 1.)
    gen = DataGenerator(config)

    data_dir = tmp_path_factory.mktemp("data")
    gt_path = data_dir / "gt.tiff"
    gen.create_groundtruth(gt_path)
    gen.create_particles(data_dir, output_extension="tiff")
    return data_dir, config


@pytest.fixture(scope="session")
def groundtruth_array(generated_particles):
    return tifffile.imread(generated_particles[0] / "gt.tiff")

@pytest.fixture(scope="session")
def particles(generated_particles):
    content = csv.reader((generated_particles[0] / "poses.csv").read_text().split('\n'))
    next(content) # skip header
    data = {}
    for row in content:
        if len(row) == 7:
            data[row[0]] = {
                "array": tifffile.imread(generated_particles[0] / row[0]),
                "rot": np.array(row[1:4], dtype=float),
                "trans": np.array(row[4:7], dtype=float)
            }
    return data


def test_generation(groundtruth_array, particles):
    assert len(particles.keys()) == N
    for k in particles:
        assert particles[k]["array"].shape == (D, D, D)
    assert groundtruth_array.shape == (D, D, D)


def test_poses(groundtruth_array: np.ndarray, particles: Dict[str, Dict[str, np.ndarray]], generated_particles: Tuple[Any, DataGenerationConfig]):
    _, config = generated_particles
    gt = groundtruth_array
    for k in particles:
        particle = particles[k]["array"]
        rot = Rotation.from_euler("ZXZ", particles[k]["rot"], degrees=True).as_matrix()
        H_rot = np.eye(4)
        H_rot[:3, :3] = rot
        H_center1 = np.eye(4)
        H_center1[:3, 3] = (np.array(particle.shape)-1) / 2
        H_center2 = np.eye(4)
        H_center2[:3, 3] = -(np.array(particle.shape)-1) / 2
        H_trans = np.eye(4)
        H_trans[:3, 3] = -particles[k]["trans"]
        H = H_trans @ H_center1 @ H_rot @ H_center2 # translation vers 0,0 -> rot -> translation vers centre -> translation
        invH = np.linalg.inv(H)
        transformed_gt = affine_transform(gt, invH, order=3)
        sigma = config.sensor.anisotropic_blur_sigma
        mode  = config.sensor.anisotropic_blur_border_mode
        transformed_gt_blurred = gaussian_filter(transformed_gt, sigma=sigma, mode=mode)

        # Is the transformed groundtruth aligned with particle ?
        (dz, dy, dx), _, _ = phase_cross_correlation(particle, transformed_gt_blurred, upsample_factor=10, disambiguate=True, normalization=None)
        assert dz <= 0.1
        assert dy <= 0.1
        assert dx <= 0.1
