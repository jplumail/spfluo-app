from .loading import isotropic_resample, resize

import argparse
import os
import glob

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--function", type=str)

    # common args
    parser.add_argument("-i", "--input", type=str, help="The image(s) to process", nargs='+')
    parser.add_argument("-o", "--output", type=str, help="The path to the output image/directory")

    # size func args
    parser.add_argument("--size", type=int)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    image_paths = list(map(os.path.abspath, args.input))
    print("Function :", args.function)
    print("Images :", image_paths)
    if args.function == "isotropic_resample":
        output_path = os.path.abspath(args.output)
        isotropic_resample(image_paths, output_path)
    if args.function == "resize":
        output_path = os.path.abspath(args.output)
        resize(image_paths, args.size, output_path)


if __name__ == '__main__':
    main(parse_args())