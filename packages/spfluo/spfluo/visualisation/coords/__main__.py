import argparse

from ..viewers import show_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="path to the image to visualize")
    parser.add_argument("--coords", type=str, required=True)
    parser.add_argument(
        "--scale", type=float, default=None, nargs="+", help="Voxel size (ZYX)"
    )
    args = parser.parse_args()

    show_points(args.file, args.coords, args.scale)


if __name__ == "__main__":
    main()
