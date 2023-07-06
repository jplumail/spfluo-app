import os
import shutil

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io


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
    io.imsave(path, np.float32(array))


def move_if_exists(src, dst):
    if os.path.exists(src):
        shutil.move(src, dst)


def make_dir(dir):
    """creates folder at location dir if i doesn't already exist"""
    if not os.path.exists(dir):
        print(f"directory {dir} created")
        os.makedirs(dir)


def delete_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


def make_dir_and_write_array(np_array, fold, name):
    make_dir(fold)
    write_array_csv(np_array, f"{fold}/{name}")
    print(f"array saved at location {fold}, ith name {name}")


def write_array_csv(np_array, path):
    pd.DataFrame(np_array).to_csv(path)


def read_csv(path, first_col=1):
    """read the csvfile at location 'path'. It reads only the colums after the column indexed by 'first_col'"""
    return np.array(pd.read_csv(path))[:, first_col:].squeeze().astype(np.float32)


def read_csvs(paths):
    contents = []
    for path in paths:
        contents.append(read_csv(path))
    return contents


def print_dictionnary_in_file(dic, file):
    """print dictionnary keys and attributes in a text file"""
    for at in list(dic.keys()):
        at_val = dic[at]
        print(f"{at} : {at_val}", file=file)


def save_figure(fold, save_name):
    make_dir(fold)
    plt.savefig(f"{fold}/{save_name}")
    print(f"figure saved at location {fold}, with name {save_name}")


if __name__ == "__main__":
    from manage_files.paths import *

    pth = f"{PATH_REAL_DATA}/Data_marine/selected_data/preprocessed_resize_ratio_2/c1"
    read_images_in_folder(pth)

    # a = np.random.random((6000, 30))
    # write_array_csv(a, './yy.csv')
