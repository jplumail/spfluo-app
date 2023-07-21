from os import PathLike
from pathlib import Path
from typing import Tuple

import numpy as np

from .config import DataGenerationConfig
from .data_generator import DataGenerator

# +------------------------------------------------------------------------------------------+ #
# |                                  GENERATE SYNTHETIC DATA                                 | #
# +------------------------------------------------------------------------------------------+ #


def init_base_config():
    config = DataGenerationConfig()
    # outliers_pass1 = Outliers(
    #    radius_range_xy=(25, 80),
    #    radius_range_z=(20, 40),
    #    nb_points_range=(1000, 3000),
    #    nb_clusters_range=(100, 150),
    #    intensity_range=(0.5, 0.7),
    # )
    # outliers_pass2 = cfg.Outliers(
    #    radius_range_xy=(150, 300),
    #    radius_range_z=(20, 40),
    #    nb_points_range=(500, 1000),
    #    nb_clusters_range=(100, 150),
    #    intensity_range=(0.5, 1.),
    # )

    # Tilt
    # outliers_pass2 = Outliers(
    #    radius_range_xy=(150, 300),
    #    radius_range_z=(20, 40),
    #    nb_points_range=(500, 1000),
    #    nb_clusters_range=(100, 200),
    #    intensity_range=(0.5, 1.0),
    # )

    # config.outliers = (outliers_pass1, outliers_pass2) # no outlier
    return config


def generate_data(
    size: int, output_dir: str, pointcloud_path: str = None, extension: str = "npz"
) -> None:
    config = init_base_config()

    # Small image
    config.target_shape = (128, 512, 512)
    config.voxelisation.num_particles = 150
    config.voxelisation.max_particle_dim = 25

    # Tilt config
    config.voxelisation.tilt_margin = 0
    config.voxelisation.tilt_strategy = "uniform"
    config.voxelisation.bandwidth = 15
    config.voxelisation.cluster_range_xy = (10, 40)
    config.voxelisation.cluster_range_z = (5, 20)
    config.voxelisation.nb_particles_per_cluster_range = (1, 1)  # no cluster
    config.augmentation.intensity_mean_ratio = 0.0  # no hole
    config.augmentation.intensity_std_ratio_range = (0.0, 0.0)  # no hole
    config.augmentation.shrink_range = (1.0, 1.0)  # no shrink

    # sensors
    config.sensor.anisotropic_blur = False
    config.sensor.poisson_noise = False
    config.sensor.gaussian_noise = False

    config.io.generated_dataset_size = size
    config.io.output_dir = output_dir
    config.io.extension = extension

    DataGenerator(config).generate_dataset()


def generate_particles(
    pointcloud_path: PathLike,
    output_dir: str,
    image_shape: int,
    num_particles: int,
    anisotropy: Tuple[float, float, float],
    target_snr: int,
):
    output_dir: Path = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    config = DataGenerationConfig()
    config.dtype = np.float32
    config.augmentation.max_translation = 0
    if pointcloud_path is not None:
        config.io.point_cloud_path = pointcloud_path
    config.io.extension = "tiff"
    config.voxelisation.image_shape = image_shape
    config.voxelisation.max_particle_dim = int(0.6 * image_shape)
    config.voxelisation.num_particles = num_particles
    config.voxelisation.bandwidth = 17
    config.sensor.anisotropic_blur_sigma = anisotropy
    config.augmentation.rotation_proba = 1
    config.augmentation.shrink_range = (1.0, 1.0)
    config.sensor.gaussian_noise_target_snr_db = target_snr
    gen = DataGenerator(config)
    gt_path = output_dir / "gt.tiff"
    gen.save_psf(output_dir / "psf.tiff")
    gen.save_groundtruth(gt_path)
    gen.create_particles(output_dir, output_extension="tiff")
