from spfluo.ab_initio_reconstruction.manage_files.read_save_files import (
    read_images_in_folder,
    save,
)
from .averaging import rotate_average

import pickle
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles_dir", type=str)
    parser.add_argument("--transformations_path", type=str)
    parser.add_argument("--average_path", type=str)
    return parser.parse_args()


def main():
    args = args_parser()
    particles, fnames = read_images_in_folder(args.particles_dir)
    with open(args.transformations_path, "rb") as f:
        trans_mat = pickle.load(f)
    t = [trans_mat[k.split(".")[0]] for k in fnames]
    avrg_particle = rotate_average(particles, t)
    save(args.average_path, avrg_particle)


if __name__ == "__main__":
    main()
