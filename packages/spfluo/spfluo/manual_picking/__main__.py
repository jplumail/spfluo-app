import argparse
import os

from .annotate import annotate


def parse_args() -> argparse.Namespace:
    """
    Arguments:
     - file: path to the input file
     - output: path to the output file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="path to the image to annotate")
    parser.add_argument("output", type=str, help="path to the output csv file")
    parser.add_argument(
        "--spacing", type=float, default=None, nargs="+", help="Voxel size (ZYX)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    p, _ = os.path.splitext(args.output)
    if args.spacing is None:
        spacing = (1.0, 1.0, 1.0)
    else:
        assert len(args.spacing) == 3
        spacing = args.spacing
    annotate(args.file, p + ".csv", spacing=spacing)


if __name__ == "__main__":
    main(parse_args())
