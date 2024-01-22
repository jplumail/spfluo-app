import torch

from spfluo.refinement.refinement import reconstruction_L2
from spfluo.utils.transform import symmetrize_poses
from spfluo.utils.volume import center_of_mass


def symmetrize(
    particle: torch.Tensor,
    center: tuple[float, float],
    symmetry: int,
    psf: torch.Tensor,
    lambda_: torch.Tensor,
):
    tensor_kwargs = dict(dtype=float, device=particle.device)
    zc, _, _ = center_of_mass(particle)
    pose_syms = symmetrize_poses(
        torch.as_tensor(
            [0, 0, 0, -(particle.shape[0] / 2 - zc), center[0], center[1]],
            **tensor_kwargs,
        ),
        symmetry=symmetry,
    )
    return reconstruction_L2(
        particle[None], psf, pose_syms[:, None], lambda_, symmetry=True
    )
