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
    parser.add_argument("--size", type=int, default=10, help="size of the box")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    p, _ = os.path.splitext(args.output)
    annotate(args.file, p + ".csv", size=args.size)


if __name__ == "__main__":
    main(parse_args())
