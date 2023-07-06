import argparse
import os

from .loading import isotropic_resample, resample, resize


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--function", type=str)

    # common args
    parser.add_argument(
        "-i", "--input", type=str, help="The image(s) to process", nargs="+"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="The path to the output image/directory"
    )

    # isotropic_resample args
    parser.add_argument(
        "--spacing", type=float, nargs="+", help="Voxel size (ZYX)", default=None
    )

    # resize args
    parser.add_argument("--size", type=int)

    # resample args
    parser.add_argument("--factor", type=float, help="Resampling factor", default=1.0)

    return parser


def main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    if args.input is None:
        parser.print_help()
        return
    image_paths = list(map(os.path.abspath, args.input))
    output_path = os.path.abspath(args.output)
    print("Function :", args.function)
    print("Images :", image_paths)
    if args.function == "isotropic_resample":
        isotropic_resample(image_paths, output_path, spacing=args.spacing)
    if args.function == "resize":
        resize(image_paths, args.size, output_path)
    if args.function == "resample":
        resample(image_paths, output_path, factor=args.factor)


if __name__ == "__main__":
    main(parse_args())
