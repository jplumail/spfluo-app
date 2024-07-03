from pathlib import Path

import numpy as np

from spfluo.picking.modules.pretraining.generate.config import (
    IO,
    Augmentation,
    DataGenerationConfig,
    Sensor,
    Voxelisation,
)
from spfluo.picking.modules.pretraining.generate.data_generator import DataGenerator

DATA_DIR = Path(__file__).parent

if __name__ == "__main__":
    np.random.seed(123)

    # Create Data Generator configs
    configs: list[DataGenerationConfig] = []
    ids: list[str] = []

    def default_config():
        D = 50
        N = 10
        pointcloud_path = DATA_DIR / "sample_centriole_point_cloud.csv"
        return DataGenerationConfig(
            dtype=np.float32,
            io=IO(
                point_cloud_path=pointcloud_path,
                extension="tiff",
            ),
            voxelisation=Voxelisation(
                num_particles=N,
                max_particle_dim=int(0.6 * D),
                image_shape=D,
                bandwidth=17,
            ),
            sensor=Sensor(gaussian_noise_target_snr_db=25),
            augmentation=Augmentation(rotation_proba=1, shrink_range=(1.0, 1.0)),
        )

    # isotropic / anisotropic datasets
    for anisotropic_param, id in [
        ((1.0, 1.0, 1.0), "isotropic-1.0"),
        ((5.0, 1.0, 1.0), "anisotropic-5.0-1.0-1.0"),
        ((10.0, 1.0, 1.0), "anisotropic-10.0-1.0-1.0"),
    ]:
        config = default_config()
        config.augmentation.max_translation = 0
        config.sensor.anisotropic_blur_sigma = anisotropic_param
        if anisotropic_param == (10.0, 1.0, 1.0):
            config.voxelisation.image_shape = 70

        configs.append(config)
        ids.append(id)

    # dataset with translations
    config = default_config()
    config.augmentation.max_translation = 100
    configs.append(config)
    ids.append("with-translations")

    # Generate datasets
    for config, id in zip(configs, ids):
        output_dir = DATA_DIR / "generated" / id
        output_dir.mkdir(exist_ok=True, parents=True)
        gen = DataGenerator(config)
        gt_path = output_dir / "gt.tiff"
        gen.save_psf(output_dir / "psf.tiff")
        gen.save_groundtruth(gt_path)
        gen.create_particles(output_dir, output_extension="tiff")
