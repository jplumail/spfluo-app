from __future__ import annotations

import tifffile

from spfluo.ab_initio_reconstruction.common_image_processing_methods.others import (
    normalize,
)
from spfluo.refinement.refinement import reconstruction_L2
from spfluo.utils.array import array_namespace, get_device, to_numpy
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
    gpu: bool = False,
    batch_size: int | None = None,
):
    assert output_volume_path.endswith(".ome.tiff")
    psf = normalize(read_image(psf_path, dtype="float64", gpu=gpu))
    xp = array_namespace(psf)
    compute_device = get_device(psf)
    psf = xp.to_device(psf, "cpu")
    host_device = get_device(psf)
    device = get_device(psf)
    particles = xp.stack(
        [
            normalize(read_image(p, dtype="float64", xp=xp, device=host_device))
            for p in particles_paths
        ]
    )
    if particles.ndim == 4:
        particles = particles[:, None]
    C = particles.shape[1]
    if psf.ndim == 3:
        psf = xp.stack((psf,) * C)
    poses, _ = read_poses(poses_paths, alphabetic_order=False)
    poses = xp.asarray(poses, dtype=xp.float64, device=host_device)
    poses = xp.permute_dims(symmetrize_poses(poses, symmetry), (1, 0, 2))
    reconstruction = xp.empty_like(particles[0])
    reconstruction = reconstruction_L2(
        particles,
        psf,
        poses,
        xp.asarray(lbda, dtype=particles.dtype, device=device),
        symmetry=True,
        device=compute_device,
        batch_size=batch_size,
        multichannel=True,
    )

    tifffile.imwrite(
        output_volume_path, to_numpy(reconstruction), metadata={"axes": "CZYX"}
    )
