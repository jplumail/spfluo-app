# This test tests ab-initio + refinement on the anisotropic dataset

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import spfluo
import spfluo.ab_initio_reconstruction.__main__ as ab_initio_main
import spfluo.refinement.__main__ as refinement_main
import spfluo.utils.__main__ as utils_main
from spfluo import data
from spfluo.ab_initio_reconstruction.learning_algorithms.gradient_descent_importance_sampling import (  # noqa: E501
    compute_energy,
)
from spfluo.utils.array import array_namespace, numpy
from spfluo.utils.loading import read_poses
from spfluo.utils.read_save_files import (
    read_image,
    read_images_in_folder,
)
from spfluo.utils.transform import get_transform_matrix
from spfluo.utils.volume import affine_transform, interpolate_to_size

if TYPE_CHECKING:
    from spfluo.utils.array import Array


def get_energy(
    reconstruction: "Array",
    psf: "Array",
    poses: "Array",
    particles: "Array",
):
    xp = array_namespace(reconstruction, psf, poses, particles)
    transforms = get_transform_matrix(
        particles[0].shape, poses[:, :3], poses[:, 3:], degrees=True
    )
    psf = interpolate_to_size(psf, particles[0].shape)
    transforms = xp.asarray(transforms, dtype=particles.dtype)
    particles_rotated = affine_transform(particles, transforms, batch=True)  # inverse
    psf_rotated = affine_transform(
        xp.stack((psf,) * particles.shape[0]), transforms, batch=True
    )

    energy = xp.mean(
        compute_energy(
            xp.fft.fftn(reconstruction),
            xp.fft.fftn(psf_rotated),
            xp.fft.fftn(particles_rotated),
        )
    )
    return energy


# works on a ~12GB VRAM GPU
@pytest.mark.slow
@pytest.mark.skipif(not spfluo.has_torch(), reason="This test requires torch installed")
@pytest.mark.skipif(
    not spfluo.has_torch_cuda(),
    reason="This test requires torch CUDA installed."
    "It would be too long without a GPU.",
)
def test_ab_initio_refinement(tmpdir):
    tmpdir = Path(tmpdir)
    generated_root_dir = Path(data.generated_anisotropic()["rootdir"])
    particles_path = generated_root_dir / "particles"
    psf_path = generated_root_dir / "psf.tiff"

    # Ab initio main
    ab_initio_parser = ab_initio_main.create_parser()
    ab_initio_args = ab_initio_parser.parse_args(
        [
            "--particles_dir",
            str(particles_path),
            "--psf_path",
            str(psf_path),
            "--output_dir",
            str(tmpdir),
            "--gpu",
            # args
            "--N_iter_max",
            str(10),
            "--eps",
            "-10",
            "--lr",
            str(0.2),
        ]
    )
    ab_initio_main.main(ab_initio_args)
    guessed_poses_path = tmpdir / "poses.csv"
    reconstruction_ab_initio_path = tmpdir / "final_recons.tif"
    assert guessed_poses_path.exists()
    assert reconstruction_ab_initio_path.exists()

    # Symmetry axis aligned with the X-axis
    poses_aligned = tmpdir / "poses_aligned.csv"
    recon_aligned = tmpdir / "recons_aligned.tiff"
    utils_parser = utils_main.create_parser()
    utils_args = utils_parser.parse_args(
        [
            "-f",
            "rotate_symmetry_axis",
            "-i",
            str(reconstruction_ab_initio_path),
            "-o",
            str(poses_aligned),
            "--poses",
            str(guessed_poses_path),
            "--rotated-volume",
            str(recon_aligned),
            "--symmetry",
            str(9),
        ]
    )
    utils_main.main(utils_args)

    # Refinement main
    refinement_parser = refinement_main.create_parser()
    reconstruction_refined_path = tmpdir / "reconstruction_refined.tiff"
    poses_refined_path = tmpdir / "poses_refined.csv"
    refinement_args = refinement_parser.parse_args(
        [
            "--particles_dir",
            str(particles_path),
            "--psf_path",
            str(psf_path),
            "--guessed_poses_path",
            str(poses_aligned),
            "--output_reconstruction_path",
            str(reconstruction_refined_path),
            "--output_poses_path",
            str(poses_refined_path),
            # refinement args
            "--symmetry",
            str(9),
            "--steps",
            "(1024,10)",
            "5",
            "5",
            "5",
            "5",
            "--ranges",
            "0",
            "40",
            "20",
            "10",
            "5",
            "-l",
            "0.001",
            "--gpu",
            "--minibatch_size",
            "256",
        ]
    )
    refinement_main.main(refinement_args)
    assert reconstruction_refined_path.exists()
    assert poses_refined_path.exists()

    # Has the energy decreased ?
    particles, _ = read_images_in_folder(str(particles_path), xp=numpy)
    psf = read_image(str(psf_path), xp=numpy)

    # Compute ab initio energy
    reconstruction_ab_initio = read_image(str(reconstruction_ab_initio_path), xp=numpy)
    poses_ab_initio, _ = read_poses(str(guessed_poses_path))
    energy_ab_initio = get_energy(
        reconstruction_ab_initio, psf, poses_ab_initio, particles
    )

    # Compute refinement energy
    reconstruction_refined = read_image(str(reconstruction_refined_path), xp=numpy)
    poses_refined, _ = read_poses(str(poses_refined_path))
    energy_refined = get_energy(reconstruction_refined, psf, poses_refined, particles)

    assert energy_ab_initio > energy_refined
