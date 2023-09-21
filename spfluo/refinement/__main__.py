import argparse

import numpy as np
import tifffile
import torch

from spfluo.ab_initio_reconstruction.manage_files.read_save_files import (
    read_image,
    read_images_in_folder,
)
from spfluo.refinement import refine
from spfluo.utils.loading import read_poses, save_poses
from spfluo.utils.log import base_parser, set_logging_level


def create_parser():
    parser = argparse.ArgumentParser(
        "Refinement",
        parents=[base_parser],
    )

    # Input files
    parser.add_argument("--particles_dir", type=str, required=True)
    parser.add_argument("--psf_path", type=str, required=True)
    parser.add_argument("--guessed_poses_path", type=str, required=True)

    # Output files
    parser.add_argument(
        "--output_reconstruction_path",
        type=str,
        required=False,
        default="./reconstruction.tiff",
    )
    parser.add_argument(
        "--output_poses_path", type=str, required=False, default="./poses.csv"
    )

    # Parameters
    def tuple_of_int(string):
        if "(" in string:
            string = string[1:-1]
        t = tuple(map(int, string.split(",")))
        if len(t) == 2:
            return t
        elif len(t) == 1:
            return t[0]
        else:
            raise TypeError

    parser.add_argument(
        "--steps", nargs="+", action="append", type=tuple_of_int, required=True
    )
    parser.add_argument(
        "--ranges", nargs="+", action="append", type=float, required=True
    )
    parser.add_argument("-l", "--lambda_", type=float, required=False, default=100.0)
    parser.add_argument(
        "--symmetry",
        type=int,
        required=False,
        default=1,
        help="Adds a constraint to the refinement. "
        "The symmetry is cylindrical around the X-axis.",
    )

    # GPU
    parser.add_argument("--gpu", action="store_true")

    return parser


def main(args):
    particles, _ = read_images_in_folder(args.particles_dir, alphabetic_order=True)
    psf = read_image(args.psf_path)
    guessed_poses, names = read_poses(args.guessed_poses_path, alphabetic_order=True)

    # Transfer to GPU
    def as_tensor(arr):
        device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
        return torch.as_tensor(
            arr.astype(np.float32), dtype=torch.float32, device=device
        )

    particles, psf, guessed_poses = map(as_tensor, (particles, psf, guessed_poses))

    reconstruction, poses = refine(
        particles,
        psf,
        guessed_poses,
        args.steps[0],
        args.ranges[0],
        args.lambda_,
        args.symmetry,
    )

    reconstruction, poses = reconstruction.cpu().numpy(), poses.cpu().numpy()
    tifffile.imwrite(args.output_reconstruction_path, reconstruction)
    save_poses(args.output_poses_path, poses, names)


if __name__ == "__main__":
    p = create_parser()
    args = p.parse_args()
    set_logging_level(args)
    main(args)
