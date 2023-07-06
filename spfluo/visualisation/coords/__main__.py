import argparse

from ..viewers import show_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="path to the image to visualize")
    parser.add_argument("--coords", type=str, required=True)
    args = parser.parse_args()

    show_points(args.file, args.coords)


if __name__ == "__main__":
    main()
