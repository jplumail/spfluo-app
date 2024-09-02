from __future__ import annotations

from typing import TYPE_CHECKING

from spfluo.refinement.refinement import reconstruction_L2
from spfluo.utils.array import array_namespace, get_device
from spfluo.utils.transform import symmetrize_poses
from spfluo.utils.volume import center_of_mass

if TYPE_CHECKING:
    from spfluo.utils.array import Array


def symmetrize(
    particle: "Array",
    center: tuple[float, float],
    symmetry: int,
    psf: "Array",
    lambda_: "Array",
):
    xp = array_namespace(particle, psf, lambda_)
    zc, _, _ = center_of_mass(particle)
    pose_syms = symmetrize_poses(
        xp.asarray(
            [0, 0, 0, -(particle.shape[0] / 2 - zc), center[0], center[1]],
            device=get_device(particle),
            dtype=xp.float64,
        ),
        symmetry=symmetry,
    )
    return reconstruction_L2(
        xp.astype(particle[None], xp.float32),
        xp.astype(psf, xp.float32),
        pose_syms[:, None],
        lambda_,
        symmetry=True,
        device=get_device(particle),
    )[0]
