from .annotate import annotate

import argparse
import os


def parse_args() -> argparse.Namespace:
    """
    Arguments:
     - file: path to the input file
     - output: path to the output file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help="path to the image to annotate")
    parser.add_argument('output', type=str, help="path to the output csv file")
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    p, _ = os.path.splitext(args.output)
    annotate(args.file, p+'.csv')

if __name__ == '__main__':
    main(parse_args())