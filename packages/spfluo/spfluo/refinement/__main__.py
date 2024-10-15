import argparse

import tifffile

from spfluo.ab_initio_reconstruction.common_image_processing_methods.others import (
    normalize,
)
from spfluo.refinement import refine
from spfluo.utils.array import array_namespace, get_device, to_numpy
from spfluo.utils.loading import read_poses, save_poses
from spfluo.utils.log import base_parser, set_logging_level
from spfluo.utils.read_save_files import (
    read_image,
    read_images_in_folder,
)


def create_parser():
    parser = argparse.ArgumentParser(
        "Refinement",
        parents=[base_parser],
    )

    # Input files
    parser.add_argument("--particles_dir", type=str, required=True)
    parser.add_argument("--psf_path", type=str, required=True)
    parser.add_argument("--guessed_poses_path", type=str, required=True)
    parser.add_argument("--initial_volume_path", type=str, required=False, default=None)
    parser.add_argument("--channel", type=int, required=False, default=None)

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

    parser.add_argument("--dtype", type=str, default="float32")

    # GPU
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--minibatch_size", type=int, default=None)

    return parser


def main(args):
    particles, _ = read_images_in_folder(
        args.particles_dir, alphabetic_order=True, gpu=args.gpu, dtype=args.dtype
    )
    xp = array_namespace(particles)
    compute_device = get_device(particles)
    particles = xp.to_device(particles, "cpu")
    host_device = get_device(particles)
    particles = xp.stack([normalize(particles[i]) for i in range(particles.shape[0])])
    if particles.ndim == 4:
        particles = particles[:, None]
    psf = normalize(
        read_image(args.psf_path, xp=xp, device=host_device, dtype=args.dtype)
    )
    if psf.ndim == 3:
        psf = xp.stack((psf,) * particles.shape[1], axis=0)
    if args.initial_volume_path:
        initial_volume = normalize(
            read_image(
                args.initial_volume_path, xp=xp, device=host_device, dtype=args.dtype
            )
        )
        if initial_volume.ndim == 3:
            initial_volume = xp.stack((initial_volume,) * particles.shape[1], axis=0)
    else:
        initial_volume = None
    guessed_poses, names = read_poses(args.guessed_poses_path, alphabetic_order=True)
    guessed_poses = xp.asarray(
        guessed_poses, device=host_device, dtype=getattr(xp, args.dtype)
    )

    if args.channel is not None:
        particles = particles[:, [args.channel]]
        psf = psf[[args.channel]]
        if initial_volume is not None:
            initial_volume = initial_volume[[args.channel]]

    reconstruction, poses = refine(
        particles,
        psf,
        guessed_poses,
        args.steps[0],
        args.ranges[0],
        initial_volume,
        args.lambda_,
        args.symmetry,
        device=compute_device,
        batch_size=args.minibatch_size,
    )

    reconstruction, poses = to_numpy(reconstruction, poses)
    tifffile.imwrite(
        args.output_reconstruction_path,
        reconstruction,
        metadata={"axes": "CZYX"},
    )
    save_poses(args.output_poses_path, poses, names)


if __name__ == "__main__":
    p = create_parser()
    args = p.parse_args()
    set_logging_level(args)
    main(args)
