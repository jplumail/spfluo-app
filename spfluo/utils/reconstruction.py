import tifffile

from spfluo.ab_initio_reconstruction.common_image_processing_methods.others import (
    normalize,
)
from spfluo.refinement.refinement import reconstruction_L2
from spfluo.utils.array import numpy as np
from spfluo.utils.loading import read_poses
from spfluo.utils.read_save_files import read_image
from spfluo.utils.transform import symmetrize_poses


def main(
    particles_paths: list[str],
    poses_paths: str,
    psf_path: str,
    output_volume_path: str,
    lbda: float,
    symmetry: int = 1,
):
    assert output_volume_path.endswith(".ome.tiff")
    psf = normalize(read_image(psf_path, dtype="float64", xp=np))
    particles = np.stack(
        [normalize(read_image(p, dtype="float64", xp=np)) for p in particles_paths]
    )
    if particles.ndim == 4:
        particles = particles[:, None]
    C = particles.shape[1]
    if psf.ndim == 3:
        psf = np.stack((psf,) * C)
    poses, _ = read_poses(poses_paths, alphabetic_order=False)
    poses = np.transpose(symmetrize_poses(poses, symmetry), (1, 0, 2))
    reconstruction = np.empty_like(particles[0])
    for c in range(C):
        reconstruction[c] = reconstruction_L2(
            particles[:, c],
            psf[c],
            poses,
            np.asarray(lbda, dtype=particles.dtype),
            symmetry=True,
            device="cpu",
        )

    tifffile.imwrite(output_volume_path, reconstruction, metadata={"axes": "CZYX"})
