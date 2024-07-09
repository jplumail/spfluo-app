import csv
import os
import shutil
from typing import List, Tuple

import numpy as np
import tifffile
from tqdm import tqdm

from ..training import make_U_mask
from ..utils import center_to_corners, load_annotations, load_array, summary

# +------------------------------------------------------------------------------------------+ #
# |                                        MAKE U MASKS                                      | #
# +------------------------------------------------------------------------------------------+ #


def make_U_masks(rootdir: str, extension: str = "npz") -> None:
    """While U masks can be made on the fly during training, it is possible to generate all
    of them upstream. This can save time when running multiples training on the same data.
    """
    print(f"Current directory: {rootdir}")
    # 1. Load images
    images_names = sorted(
        list(filter(lambda x: x.endswith(extension), os.listdir(rootdir)))
    )
    images_paths = sorted(list(map(lambda x: os.path.join(rootdir, x), images_names)))
    images = [load_array(path) for path in tqdm(images_paths, desc="Load images")]
    # 2. Make U masks
    U_masks = [make_U_mask(image) for image in tqdm(images, desc="Create masks")]
    # 3. Save U masks
    output_dir = os.path.join(rootdir, "U_masks")
    os.makedirs(output_dir, exist_ok=True)
    for name, u_mask in tqdm(
        zip(images_names, U_masks), total=len(U_masks), desc="Save masks"
    ):
        tifffile.imwrite(os.path.join(output_dir, name), u_mask)


# +------------------------------------------------------------------------------------------+ #
# |                  RANDOM (NEGATIVE) AND AROUND PARTICLES (POSITIVE) CROPS                 | #
# +------------------------------------------------------------------------------------------+ #

DEFAULT_NEGATIVE_CENTERS = 8


def generate_random_negative_centers(
    positive_centers: np.ndarray,
    crop_size: Tuple[int],
    shape: Tuple[int],
    pos_ratio: float,
) -> np.ndarray:
    num_negative_centers = int(len(positive_centers) / pos_ratio)
    if num_negative_centers < DEFAULT_NEGATIVE_CENTERS:
        num_negative_centers = DEFAULT_NEGATIVE_CENTERS
    negative_centers = np.array([0, 0, 0])
    while len(negative_centers) <= num_negative_centers:
        current_centers = np.vstack((negative_centers, positive_centers))
        border = np.array(crop_size) // 2
        low, high = border, np.array(shape) - border
        new_center = np.random.randint(low, high=high)
        center_tile = np.repeat(new_center[np.newaxis, :], len(current_centers), axis=0)
        distances = np.linalg.norm(center_tile - current_centers.astype(float), axis=1)
        if distances.min() > max(crop_size):
            negative_centers = np.vstack((negative_centers, new_center))
    return negative_centers[1:]


def random_shift_center(
    center: Tuple[int],
    crop_size: Tuple[int],
    max_size: Tuple[int],
    margin: Tuple[int],
) -> Tuple[int]:
    random_shifts = []
    for s, c, m, M in zip(crop_size, center, margin, max_size):
        inner_range = s // 2 - M // 2 - m
        shift_max = min(inner_range, M - c - s // 2)
        shift_min = max(-inner_range, -c + s // 2)
        if shift_min >= shift_max:
            random_shifts.append(0)
        else:
            random_shifts.append(np.random.randint(shift_min, shift_max))
    new_center = [c + s for c, s in zip(center, random_shifts)]
    return new_center


def reframe_corners_if_needed(
    corners: Tuple[int], crop_size: Tuple[int], max_size: Tuple[int]
) -> Tuple[int]:
    d, h, w = crop_size
    D, H, W = max_size
    x_min, y_min, z_min, x_max, y_max, z_max = corners
    z_min, y_min, x_min = max(0, z_min), max(0, y_min), max(0, x_min)
    z_max, y_max, x_max = z_min + d, y_min + h, x_min + w
    z_max, y_max, x_max = min(D, z_max), min(H, y_max), min(W, x_max)
    z_min, y_min, x_min = z_max - d, y_max - h, x_max - w
    return x_min, y_min, z_min, x_max, y_max, z_max


def crop_one_particle(
    image: np.ndarray,
    center: np.ndarray,
    crop_size: Tuple[int],
    max_size: Tuple[int],
    margin: Tuple[int],
    random_shift: bool,
) -> np.ndarray:
    if random_shift:
        center = random_shift_center(center, crop_size, max_size, margin)
    corners = center_to_corners(center, crop_size)
    corners = reframe_corners_if_needed(corners, crop_size, max_size)
    x_min, y_min, z_min, x_max, y_max, z_max = corners
    return image[z_min:z_max, y_min:y_max, x_min:x_max]


def save(
    image: np.ndarray,
    image_name: str,
    crop_index: str,
    output_dir: str,
    extension: str = "npz",
) -> None:
    image_prefix = os.path.splitext(image_name)[0]
    name = f"{image_prefix}_{crop_index}"
    output = os.path.join(output_dir, name)
    if extension == "npz":
        np.savez(output + ".npz", image=image)
    elif extension == "tiff" or extension == "tif":
        image = image.astype(float)
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)
        tifffile.imwrite(output + ".tiff", image)


def crop(
    rootdir: str,
    output_dir: str,
    crop_size: Tuple[int],
    margin: Tuple[int],
    pos_ratio: float = 1.0,
    positive_only: bool = False,
    extension: str = "npz",
    csv_name: str = "train_coordinates",
) -> None:
    """Crop around particles and save cropped images with respect to a given ratio of positive
    samples.

    Args:
        rootdir (str): Path to npz containing 3d images and masks to be cropped and a csv with
                       particles center coordinates.
        crop_size (Optional, Tuple[int]): (depth, height, width).
        margin (Optional, Tuple[int]): When randomly shifting bbox, on each axis the shift
                                        range will be (size // 2 - max_particle_dim // 2 - margin),
                                        and when generating random negative centers, they will be
                                        at least margin away from image border.
        pos_ratio (Optional, float): Will save N positive crops and N / ratio negative crops.
                                     In [0, 1].
        positive_only (Optional, bool): Wether or not to generate positive crop only. Usefull for
                                        tilt learning. Default to False.
        csv_name (Optional, str): Name of the annotations csv to load.
                                 Defaults to 'train_coordinates'.
    """
    positive_dir = os.path.join(rootdir, output_dir, "positive")
    negative_dir = os.path.join(rootdir, output_dir, "negative")
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)
    images_list = sorted(
        list(filter(lambda x: x.endswith(extension), os.listdir(rootdir)))
    )
    annotations = load_annotations(os.path.join(rootdir, f"{csv_name}.csv"))
    print(f"Current directory: {rootdir}")
    print(
        f"Found {len(images_list)} images with {len(annotations)} particles annotated."
    )
    if len(annotations) > 0:
        for image_name in tqdm(images_list):
            image = load_array(os.path.join(rootdir, image_name))
            centers = annotations[annotations[:, 0] == image_name, 2:].astype(float)
            indices = annotations[annotations[:, 0] == image_name, 1]
            centers = np.rint(centers).astype(int)
            for crop_index, center in zip(indices, centers):
                crop_args = [image, center, crop_size, image.shape, margin]
                cropped_image = crop_one_particle(
                    *crop_args, random_shift=False
                )  # was True
                save(cropped_image, image_name, crop_index, positive_dir, extension)
            if not positive_only:
                args = [centers, crop_size, image.shape, pos_ratio]
                negative_centers = generate_random_negative_centers(*args)
                for crop_index, center in enumerate(negative_centers):
                    crop_args = [image, center, crop_size, image.shape, margin]
                    cropped_image = crop_one_particle(*crop_args, random_shift=False)
                    save(cropped_image, image_name, crop_index, negative_dir, extension)


# +------------------------------------------------------------------------------------------+ #
# |                              TRAIN/VAL/TEST SPLITS AND PATHS                             | #
# +------------------------------------------------------------------------------------------+ #


def train_test_split(
    rootdir: str, images: List[str], image_format: str = "npz", split: float = 0.8
) -> None:
    """Creates train/ and test/ rootdir's subfolders and put into them splitted images
        and annotations.

    Args:
        rootdir (str): Folder containing the images to split and the annotations csv.
        image_format (str, optional): Image file format. Defaults to 'npz'.
        split (float, optional): Ratio len(train) / len(test). Defaults to 0.8.
    """
    train_length = int(split * len(images))
    train_images, test_images = images[:train_length], images[train_length:]
    train_dir, test_dir = os.path.join(rootdir, "train"), os.path.join(rootdir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for image in train_images:
        shutil.copyfile(os.path.join(rootdir, image), os.path.join(train_dir, image))
    for image in test_images:
        shutil.copyfile(os.path.join(rootdir, image), os.path.join(test_dir, image))
    annotations = load_annotations(os.path.join(rootdir, "coordinates.csv"))
    train_annotations = []
    for image in train_images:
        train_annotations.extend(annotations[annotations[:, 0] == image])
    train_annotations = np.array(train_annotations, dtype=object)
    csv_path = os.path.join(train_dir, "train_coordinates.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(train_annotations)
    test_annotations = []
    for image in test_images:
        test_annotations.extend(annotations[annotations[:, 0] == image])
    test_annotations = np.array(test_annotations, dtype=object)
    csv_path = os.path.join(test_dir, "test_coordinates.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(test_annotations)

    return train_images


def train_val_split(
    traindir: str, images: List[str], image_format: str = "npz", split: float = 0.8
) -> None:
    """After a train/test split has occured, an other train/val split is required inside the
    train subfolder. This could be done on the fly during training but hard separating the files
    seems cleaner and force a constant validation set regardless of the training config.

    Args:
        traindir (str): The train subfolder.
        image_format (str, optional): Used to detect images inside the given folder.
                                      Defaults to 'npz'.
        split (float, optional): Proportion of images to keep for training. Defaults to 0.8.
    """
    train_length = int(split * len(images))
    train_images, val_images = images[:train_length], images[train_length:]
    parent_dir = os.path.dirname(os.path.abspath(traindir))
    val_dir = os.path.join(parent_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    for image in val_images:
        os.replace(os.path.join(traindir, image), os.path.join(val_dir, image))
    annotations = load_annotations(os.path.join(traindir, "train_coordinates.csv"))
    train_annotations = []
    for image in train_images:
        train_annotations.extend(annotations[annotations[:, 0] == image])
    train_annotations = np.array(train_annotations, dtype=object)
    csv_path = os.path.join(traindir, "train_coordinates.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(train_annotations)
    val_annotations = []
    for image in val_images:
        val_annotations.extend(annotations[annotations[:, 0] == image])
    val_annotations = np.array(val_annotations, dtype=object)
    csv_path = os.path.join(val_dir, "val_coordinates.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(val_annotations)


def train_val_split2(
    traindir: str, images: List[str], image_format: str = "npz", split: float = 0.8
) -> None:
    """After a train/test split has occured, an other train/val split is required inside the
    train subfolder. This could be done on the fly during training but hard separating the files
    seems cleaner and force a constant validation set regardless of the training config.

    Args:
        traindir (str): The train subfolder.
        image_format (str, optional): Used to detect images inside the given folder.
                                      Defaults to 'npz'.
        split (float, optional): Proportion of images to keep for training. Defaults to 0.8.
    """
    parent_dir = os.path.dirname(os.path.abspath(traindir))
    val_dir = os.path.join(parent_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    for image in images:
        shutil.copyfile(os.path.join(traindir, image), os.path.join(val_dir, image))
    annotations = load_annotations(os.path.join(traindir, "train_coordinates.csv"))
    train_length = int(split * len(annotations))
    train_annotations = annotations[:train_length]
    val_annotations = annotations[train_length:]
    train_annotations = np.array(train_annotations, dtype=object)
    csv_path = os.path.join(traindir, "train_coordinates.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(train_annotations)
    val_annotations = np.array(val_annotations, dtype=object)
    csv_path = os.path.join(val_dir, "val_coordinates.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(val_annotations)


def train_val_test_splits(
    rootdir: str,
    images_names: List[str],
    train_test_ratio: float,
    train_val_ratio: float,
    image_format: str = "npz",
) -> None:
    train_images = train_test_split(
        rootdir, images_names, image_format, train_test_ratio
    )
    # train_val_split(os.path.join(rootdir, 'train'), train_images, image_format, train_val_ratio)
    train_val_split2(
        os.path.join(rootdir, "train"), train_images, image_format, train_val_ratio
    )


# +------------------------------------------------------------------------------------------+ #
# |                                       EXTERNAL CALL                                      | #
# +------------------------------------------------------------------------------------------+ #


def prepare(
    rootdir: str,
    extension: str,
    pointcloud_path: str = None,
    size: int = None,
    make_u_masks: bool = False,
    train_test_split: float = None,
    train_val_split: float = None,
    crop_output_dir: str = None,
    crop_size: Tuple[int, ...] = None,
    margin: Tuple[int, ...] = 0,
    pos_ratio: float = 1,
    positive_only: bool = False,
    downscale: float = 1.0,
) -> None:
    """Being given a path to a directory, prepare the image for training by doing any
    combinations of the following operations:
    1. generate synthetic data
    2. make u masks
    3. train/val/test splits
    4. make crops

    Args:
        rootdir (str): Directory which (will) contain images and annotations.
        extension (str, optional): Images format.
        size (int, optional): Number of images to generate.
        make_u_masks (bool): If True, will generate one U mask per image in a subfolder
                             rootdir/U_masks/. Defaults to False.
        train_test_split (float, optional): First, the (train+val)/test split will occur.
                                            This determines the train/test number of images ratio.
                                            Defaults to None.
        train_val_split (float, optional): After the (train+val)/test split, the (train+val)
                                           images will be splitted again. Defaults to None.

        crop_size (Tuple[int], optional): Defaults to (64, 128, 128).
        max_particle_dim (int, optional): Defaults to 50.
        margin (Tuple[int], optional): Crop will be randomly shifted so that they are not always
                                       centered on a particle. The margin controls how close to a
                                       border the particle can be after the shift.
                                       Defaults to (0, 0, 0).
        pos_ratio (float, optional): Positive/Negative crop ratio. Defaults to 1..
        positive_only (bool, optional): Crop only positive. Defaults to False.
        downscale (float, optional): Downscale factor. Default to 1 (no downscale)
    """
    summary(locals(), "DATA PREPARATION")
    # STEP 1: Data generation
    print("\nPreparing data for training ...\n")
    if size is not None:
        print("| Generating dataset ...")
        try:
            from .generate import generate_data
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                str(e)
                + "\n\nCannot generate synthetic data, you may need the pyfigtree module for that. \
Try `pip install spfluo[figtree]` or `pip install pyfigtree`."
            )
        generate_data(size, rootdir, pointcloud_path, extension)
    # STEP 2: Train/Val/Test splits
    if train_test_split is not None and train_val_split is not None:
        images_names = list(
            filter(lambda x: x.endswith(extension), os.listdir(rootdir))
        )
        # images_names = [str(i)+'.'+extension for i in range(size)]
        print("| Splitting data into train, val, and test subsets ...")
        train_val_test_splits(
            rootdir, images_names, train_test_split, train_val_split, extension
        )
    # STEP 3: Make U masks
    if make_u_masks:
        print("| Creating U masks ...")
        make_U_masks(os.path.join(rootdir, "train"), extension)
        make_U_masks(os.path.join(rootdir, "val"), extension)
    # STEP 4: Make crops
    if crop_output_dir is not None:
        print("| Generating crops inside images ...")
        crop_kwargs = {
            "crop_size": tuple(np.rint(np.array(crop_size) / downscale).astype(int)),
            "margin": margin,
            "pos_ratio": pos_ratio,
            "positive_only": positive_only,
            "extension": extension,
        }
        for d in ["train", "val", "test"]:
            crop_kwargs["csv_name"] = d + "_coordinates"
            subdir = os.path.join(rootdir, d)
            if os.path.exists(subdir):
                crop(subdir, crop_output_dir, **crop_kwargs)
            else:
                print(f"{subdir} not found")
