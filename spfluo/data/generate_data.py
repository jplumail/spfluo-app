from pathlib import Path
from typing import Tuple

import numpy as np

from spfluo.picking.modules.pretraining.generate.generate_data import generate_particles

D = 50
N = 10
DATA_DIR = Path(__file__).parent
pointcloud_path = DATA_DIR / "sample_centriole_point_cloud.csv"


def get_ids(anisotropy: Tuple[float, float, float]):
    dz, dy, dx = anisotropy
    assert dx == dy
    if dz == dy:
        return "isotropic-" + str(dz)
    else:
        return "anisotropic-" + str(dz) + "-" + str(dy) + "-" + str(dx)


if __name__ == "__main__":
    for anisotropic_param in [(1.0, 1.0, 1.0), (5.0, 1.0, 1.0)]:
        id = get_ids(anisotropic_param)
        root_dir: Path = DATA_DIR / "generated" / id
        root_dir.mkdir(exist_ok=True, parents=True)
        np.random.seed(123)
        generate_particles(pointcloud_path, root_dir, D, N, anisotropic_param, 20)
