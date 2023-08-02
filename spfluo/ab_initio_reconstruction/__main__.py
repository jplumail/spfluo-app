import argparse

import numpy as np

from spfluo.ab_initio_reconstruction.api import AbInitioReconstruction
from spfluo.ab_initio_reconstruction.manage_files.read_save_files import (
    read_image,
    read_images_in_folder,
)
from spfluo.utils.volume import interpolate_to_size


def create_parser():
    parser = argparse.ArgumentParser("Ab initio reconstruction")

    # Input files
    parser.add_argument("--particles_dir", type=str)
    parser.add_argument("--psf_path", type=str)

    # Ab initio params
    parser.add_argument("--M_axes", type=int, default=360**2)
    parser.add_argument("--M_rot", type=int, default=360)
    parser.add_argument("--dec_prop", type=float, default=1.2)
    parser.add_argument("--init_unif_prop", nargs=2, type=float, default=(1, 1))
    parser.add_argument("--coeff_kernel_axes", type=float, default=50.0)
    parser.add_argument("--coeff_kernel_rot", type=float, default=5.0)
    parser.add_argument("--eps", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--N_axes", type=int, default=25)
    parser.add_argument("--N_rot", type=int, default=20)
    parser.add_argument("--prop_min", type=float, default=0)
    parser.add_argument("--interp_order", type=int, default=3)
    parser.add_argument("--N_iter_max", type=int, default=20)
    parser.add_argument("--gaussian_kernel", type=bool, default=True)
    parser.add_argument("--N_iter_with_unif_distr", type=int, default=None)
    parser.add_argument("--epochs_of_suppression", type=int, default=None)
    parser.add_argument("--proportion_of_views_suppressed", type=float, default=None)
    parser.add_argument("--convention", type=str, default="XZX")
    parser.add_argument("--dtype", type=type, default=np.float64)
    parser.add_argument("--reg_coeff", type=float, default=0)
    parser.add_argument("--beta_sampling", type=float, default=0)
    parser.add_argument("--epoch_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beta_grad", type=float, default=0)
    parser.add_argument("--random_sampling", type=bool, default=False)

    # GPU acceleration
    parser.add_argument("--gpu", type=str, default=None)

    # Ouput directory
    parser.add_argument("--output_dir", type=str)

    return parser


def main(args):
    particles, _ = read_images_in_folder(args.particles_dir)
    psf = read_image(args.psf_path)
    ab_initio_params = dict(vars(args))
    for k in ["output_dir", "gpu", "psf_path", "particles_dir"]:
        ab_initio_params.pop(k)
    reconstruction = AbInitioReconstruction(**ab_initio_params)
    psf = interpolate_to_size(psf, particles.shape[1:])
    reconstruction.fit(particles, psf=psf, gpu=args.gpu, output_dir=args.output_dir)


if __name__ == "__main__":
    main(create_parser().parse_args())
