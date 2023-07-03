import pathlib
import numpy as np
from spfluo.picking.modules.pretraining.generate.data_generator import DataGenerationConfig, DataGenerator
from scipy.ndimage import affine_transform
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


# TODO: should test the opposite, transforming the GT into the particles to see if they match
def test_poses(groundtruth_array, particles):
    gt = groundtruth_array
    for k in particles:
        rot = Rotation.from_euler("ZXZ", particles[k]["rot"], degrees=True).as_matrix()
        H_rot = np.eye(4)
        H_rot[:3, :3] = rot
        H_center1 = np.eye(4)
        H_center1[:3, 3] = (np.array(particles[k]["array"].shape)-1) / 2
        H_center2 = np.eye(4)
        H_center2[:3, 3] = -(np.array(particles[k]["array"].shape)-1) / 2
        H_trans = np.eye(4)
        H_trans[:3, 3] = -particles[k]["trans"]
        H = H_trans @ H_center1 @ H_rot @ H_center2 # translation vers 0,0 -> rot -> translation vers centre -> translation
        transformed_particle = affine_transform(particles[k]["array"], H, order=5)

        # Is the transformed particle aligned with groundtruth ?
        (dz, dy, dx), _, _ = phase_cross_correlation(gt, transformed_particle, upsample_factor=10, disambiguate=True, normalization=None)
        assert dy < 0.5 and dx < 0.5