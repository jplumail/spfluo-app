import argparse
import logging
import os

from spfluo.utils.log import base_parser, set_logging_level

from .loading import isotropic_resample, resample, resize
from .rotate_symmetry_axis import main as rotate_symmetry_axis_main

utils_logger = logging.getLogger("spfluo.utils")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Utils functions", parents=[base_parser])
    parser.add_argument("-f", "--function", type=str)

    # common args
    parser.add_argument(
        "-i", "--input", type=str, help="The image(s) to process", nargs="+"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The path to the output image/directory",
        default=None,
    )

    # isotropic_resample args
    parser.add_argument(
        "--spacing", type=float, nargs="+", help="Voxel size (ZYX)", default=None
    )

    # resize args
    parser.add_argument("--size", type=int)

    # resample args
    parser.add_argument("--factor", type=float, help="Resampling factor", default=1.0)

    # rotate_symmetry_axis args
    parser.add_argument("--symmetry", type=int, help="symmetry degree of the particle")
    parser.add_argument(
        "--convention", type=str, help="scipy rotation convention", default="XZX"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="threshold in percentage of the max",
        default=0.5,
    )
    parser.add_argument("--poses", type=str, help="path to poses", default=None)
    parser.add_argument(
        "--rotated-volume", type=str, help="path to the rotated volume", default=None
    )

    return parser


def main(args: argparse.Namespace) -> None:
    if args.input is None:
        parser.print_help()
        return
    image_paths = list(map(os.path.abspath, args.input))
    output_path = os.path.abspath(args.output) if args.output else None
    utils_logger.info("Function : " + args.function)
    utils_logger.debug("Images :" + str(image_paths))
    if args.function == "isotropic_resample":
        isotropic_resample(image_paths, output_path, spacing=args.spacing)
    if args.function == "resize":
        resize(image_paths, args.size, output_path)
    if args.function == "resample":
        resample(image_paths, output_path, factor=args.factor)
    if args.function == "rotate_symmetry_axis":
        rotate_symmetry_axis_main(
            args.input,
            args.symmetry,
            args.convention,
            args.threshold,
            args.rotated_volume,
            args.poses,
            args.output,
        )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    set_logging_level(args)
    main(args)
