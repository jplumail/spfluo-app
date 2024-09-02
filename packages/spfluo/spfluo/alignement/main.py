import argparse
import csv

import numpy as np
import scipy.ndimage as ndi

from .get_poses import (
    iterative_suppression,
    load_patches,
    load_views,
    normalize_patches,
)


def parse_args():
    parser = argparse.ArgumentParser("Setup pipeline")

    parser.add_argument("--crop_dir", type=str, default=None)
    parser.add_argument("--views", type=str, default=None)

    parser.add_argument("--number_side", type=int, default=None)
    parser.add_argument("--number_top", type=int, default=None)

    parser.add_argument("--poses", type=str, default=None)
    return parser.parse_args()


def main(args):
    views, image_names = load_views(args.views)
    patches = load_patches(args.crop_dir, image_names)

    patches_2D = patches.sum(axis=1)
    patches_2D = normalize_patches(patches_2D)
    (
        mask_keep_side,
        params_side,
        avrg_particle_side,
        variance_map_side,
    ) = iterative_suppression(patches_2D[views == 1], args.number_side)
    (
        mask_keep_top,
        params_top,
        avrg_particle_top,
        variance_map_top,
    ) = iterative_suppression(patches_2D[views == 0], args.number_top)

    params = np.zeros(
        (patches.shape[0], 6)
    )  # shape (N, (euler1, euler2, euler3, tz, ty, tx))
    mask_keep = np.zeros((patches.shape[0]), dtype=bool)

    # Side params + trouver l'angle absolu
    params[views == 1, [0, 4, 5]] = params_side
    params[views == 1, 1] = 90
    mask_keep[views == 1] = mask_keep_side

    # Top params
    params[views == 0, [2, 4, 5]] = params_top
    mask_keep[views == 0] = mask_keep_top

    # Z-axis translation
    # Mass center in the center of the patch
    centers_of_mass = np.array(list(map(lambda x: ndi.center_of_mass(x)[0], patches)))
    params[:, 3] = -centers_of_mass

    with open(args.poses, "w") as f:
        csv.writer(f).writerows(params)


if __name__ == "__main__":
    main(parse_args())
