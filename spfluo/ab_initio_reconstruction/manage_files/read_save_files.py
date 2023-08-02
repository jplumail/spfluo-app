import os

import imageio
import numpy as np
import pandas as pd
import tifffile


def read_image(path):
    return np.array(imageio.mimread(path, memtest=False))


def read_images_in_folder(fold, alphabetic_order=True):
    """read all the images inside folder fold"""
    files = os.listdir(fold)
    if alphabetic_order:
        files = sorted(files)
    images = []
    for fn in files:
        pth = f"{fold}/{fn}"
        im = read_image(pth)
        images.append(im)
    return np.array(images), files


def save(path, array):
    # save with conversion to float32 so that imaej can open it
    tifffile.imwrite(path, np.float32(array))


def make_dir(dir):
    """creates folder at location dir if i doesn't already exist"""
    if not os.path.exists(dir):
        print(f"directory {dir} created")
        os.makedirs(dir)


def make_dir_and_write_array(np_array, fold, name):
    make_dir(fold)
    write_array_csv(np_array, f"{fold}/{name}")
    print(f"array saved at location {fold}, ith name {name}")


def write_array_csv(np_array, path):
    pd.DataFrame(np_array).to_csv(path)
