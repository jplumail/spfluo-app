import argparse
import csv
import os

import numpy as np
import tifffile
import torch

from spfluo.ab_initio_reconstruction.manage_files.read_save_files import (
    read_image,
    read_images_in_folder,
)
from spfluo.refinement import refine


def read_poses(path: str, alphabetic_order=True):
    content = csv.reader(open(path, "r").read().split("\n"))
    next(content)
    poses, fnames = [], []
    for row in content:
        if len(row) > 0:
            poses.append(np.array(row[1:], dtype=float))
            fnames.append(os.path.basename(row[0]))
    if alphabetic_order:
        _, poses = zip(*sorted(zip(fnames, poses), key=lambda x: x[0]))
    poses = np.stack(poses)
    return poses


def save_poses(path: str, poses: np.ndarray):
    with open(path, "w") as f:
        f.write("name,rot1,rot2,rot3,t1,t2,t3\n")
        for p in poses:
            pose = list(map(str, p.tolist()))
            f.write(",".join([""] + pose))
            f.write("\n")


def create_parser():
    parser = argparse.ArgumentParser("Refinement")

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
    parser.add_argument("--symmetry", type=int, required=False, default=1)

    return parser


def main(args):
    particles, _ = read_images_in_folder(args.particles_dir, alphabetic_order=True)
    psf = read_image(args.psf_path)
    guessed_poses = read_poses(args.guessed_poses_path, alphabetic_order=True)

    # Transfer to GPU
    def as_tensor(arr):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

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
    save_poses(args.output_poses_path, poses)


if __name__ == "__main__":
    main(create_parser().parse_args())
