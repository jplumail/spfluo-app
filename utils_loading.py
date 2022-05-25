import os
import pickle
import random
import imageio
import numpy as np
from typing import Tuple, Dict
import torch


def load_data(rootdir):

    views, patches_names = load_views(os.path.join(rootdir, "views.csv"))
    patches = load_patches(os.path.join(rootdir, "test/cropped/positive"), patches_names).astype(float)

    coords = load_annotations(os.path.join(rootdir, 'coordinates.csv'))[:,2:].astype(float)
    angles = load_angles(os.path.join(rootdir, 'angles.csv'))[:,2:].astype(float)
    translations = coords - np.rint(coords)
    poses = np.concatenate([angles, translations], axis=1).astype(float)

    psf = load_array(os.path.join(rootdir, 'psf.tif'))

    return patches, coords, angles, views, poses, psf


def seed_all(seed_numpy: bool=True) -> None:
    """ For reproductibility.

    But, if we shuffle annotations between runs, we want each shuffling to be unique.
    Hence the option to choose if numpy must be seeded.
    """
    random.seed(0)
    torch.manual_seed(0)
    if seed_numpy:
        np.random.seed(0)


def load_array(path: str) -> np.ndarray:
    """ Takes a complete path to a file and return the corresponding numpy array.

    Args:
        path (str): Path to the file to read. It MUST contains the extension as this will
                    determines the way the numpy array is loaded from the file.

    Returns:
        np.ndarray: The array stored in the file described by 'path'.
    """
    extension = os.path.splitext(path)[-1]
    if extension == '.npz':
        return np.load(path)['image']
    elif extension in ['.tif', '.tiff']:
        image = np.stack(imageio.mimread(path, memtest=False)).astype(np.int16)
        # Some tiff images are heavily imbalanced: their data type is int16 but very few voxels
        # are actually > 255. If this is the case, the image in truncated and casted to uint8.
        if image.dtype == np.int16 and ((image > 255).sum() / image.size) < 1e-3:
            image[image > 255] = 255
            image = image.astype(np.uint8)
        return image
    error_msg = f'Found extension {extension}. Extension must be one of npz or tif.'
    raise NotImplementedError(error_msg)


def load_annotations(csv_path: str) -> np.ndarray:
    """ Csv containing coordinates of objects center.

    Args:
        csv_path (str): Path of the file to read.

    Returns:
        np.ndarray: Array into which each line is alike ('image name', particle_id, z, y, x).
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line_data = line.split(',')
        line_data[2:] = list(map(float, line_data[2:]))
        data.append(line_data)
    return np.array(data, dtype=object)

def load_angles(csv_path: str) -> np.ndarray:
    """ Csv containing angles of particles.

    Args:
        csv_path (str): Path of the file to read.

    Returns:
        np.ndarray: Array into which each line is alike ('image name', particle_id, z, y, x).
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line_data = line.split(',')
        line_data[2:] = list(map(float, line_data[2:]))
        data.append(line_data)
    return np.array(data, dtype=object)


def load_views(views_path):
    _, ext = os.path.splitext(views_path)
    if ext == '.csv': # DONT USE CSV NOT WORKING 
        views_ = load_annotations(views_path)
        views = views_[:,2].astype(int)
        patches_names = np.array([
            os.path.splitext(im_name)[0] + '_' + patch_index + os.path.splitext(im_name)[1]
            for im_name, patch_index in views_[:,[0,1]]
        ])
    elif ext == '.pickle':
        with open(views_path, 'rb') as f:
            views_ = pickle.load(f)
        views = np.concatenate([views_[image_name][0] for image_name in views_.keys()])
        patches_names = np.concatenate([views_[image_name][2] for image_name in views_.keys()])
    
    return views, patches_names


def load_patches(crop_dir, patches_names):
    patches = [load_array(os.path.join(crop_dir,p)) for p in patches_names] # (N,z,y,x)
    return np.stack(patches, axis=0)


def center_to_corners(center: Tuple[int], size: Tuple[int]) -> Tuple[int]:
    depth, height, width = size
    center_z, center_y, center_x = center
    z_min = center_z - depth // 2
    y_min = center_y - height // 2
    x_min = center_x - width // 2
    z_max = z_min + depth
    y_max = y_min + height
    x_max = x_min + width
    return x_min, y_min, z_min, x_max, y_max, z_max


def summary(kwargs: Dict, title: str, output: str=None, return_table: bool=False) -> None:
    # 1. Get key max length
    key_length = max([len(k) for k in kwargs.keys()])
    # 2. Get total length
    length = max([key_length + len(str(v)) for k, v in kwargs.items()]) + 4
    # 3. Define table delimiter and title
    hbar = '+' + length * '-' + '+' + '\n'
    pad_left  = (length - len(title)) // 2
    pad_right = length - len(title) - pad_left
    title = '|' + pad_left * ' ' + title + pad_right * ' ' + '|' + '\n'
    # 4. Create table string
    table = '\n' + hbar + title + hbar
    # 5. Add lines to table
    for k, v in kwargs.items():
        line = k + (key_length - len(k)) * '.' + ': ' + str(v)
        line = '| ' + line + (length - 1 - len(line)) * ' ' + '|' + '\n'
        table += line
    table += hbar
    # 6. Print table
    print(table)
    # 7. Save table to txt file (optional)
    if output is not None:
        with open(output, 'w') as file:
            file.write(table)
    # 8. Return table (optional)
    if return_table:
        return table


def send_mail(subject: str, body: str) -> None:
    import smtplib
    import ssl
    port     = 465  # SSL
    password = "hackitifyouwantidontcare"
    sender_email   = "icuberemotedev@gmail.com"
    receiver_email = "vedrenneluc@gmail.com"
    message = f"""\
    {subject}

    {body}
    """
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

