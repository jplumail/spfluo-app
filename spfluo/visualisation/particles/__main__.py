import argparse
import os

from ..viewers import show_particles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="+", help="paths to the particles")
    args = parser.parse_args()

    image_paths = list(map(os.path.abspath, args.files))

    show_particles(image_paths)


if __name__ == "__main__":
    main()
