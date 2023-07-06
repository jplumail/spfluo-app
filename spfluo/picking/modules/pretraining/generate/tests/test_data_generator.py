from typing import Any, Dict, Tuple
import numpy as np
from spfluo.picking.modules.pretraining.generate.data_generator import DataGenerationConfig, DataGenerator
from spfluo.utils.transform import get_transform_matrix
from scipy.ndimage import affine_transform, convolve
from scipy.spatial.transform import Rotation
from skimage.registration import phase_cross_correlation
import pytest
import tifffile
import csv


D = 50
N = 20

@pytest.fixture(scope="session")
def generated_particles_dir(tmp_path_factory):
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
    gen.save_psf(data_dir / "psf.tiff")
    gen.save_groundtruth(gt_path)
    gen.create_particles(data_dir, output_extension="tiff")
    return data_dir

@pytest.fixture(scope="session")
def psf_array(generated_particles_dir):
    return tifffile.imread(generated_particles_dir / "psf.tiff")

@pytest.fixture(scope="session")
def groundtruth_array(generated_particles_dir):
    return tifffile.imread(generated_particles_dir / "gt.tiff")

@pytest.fixture(scope="session")
def particles(generated_particles_dir):
    content = csv.reader((generated_particles_dir / "poses.csv").read_text().split('\n'))
    next(content) # skip header
    data = {}
    for row in content:
        if len(row) == 7:
            data[row[0]] = {
                "array": tifffile.imread(generated_particles_dir / row[0]),
                "rot": np.array(row[1:4], dtype=float),
                "trans": np.array(row[4:7], dtype=float)
            }
    return data


def test_generation(psf_array, groundtruth_array, particles):
    assert len(particles.keys()) == N
    for k in particles:
        assert particles[k]["array"].shape == (D, D, D)
    assert groundtruth_array.shape == (D, D, D)


def test_poses(psf_array: np.ndarray, groundtruth_array: np.ndarray, particles: Dict[str, Dict[str, np.ndarray]]):
    gt = groundtruth_array
    for k in particles:
        particle = particles[k]["array"]

        # H go from gt to particle (which is transformed)
        H = get_transform_matrix(particle.shape, particles[k]["rot"], particles[k]["trans"], convention="ZXZ", degrees=True)
        
        # invert this because scipy's affine_transform works backward
        invH = np.linalg.inv(H)
        transformed_gt = affine_transform(gt, invH, order=1)

        # Apply the data model
        transformed_gt_blurred = convolve(transformed_gt, psf_array, mode='constant', cval=0.)

        # Is the transformed groundtruth aligned with particle ?
        assert are_volumes_aligned(particle, transformed_gt_blurred)