"""This file defines several routines required for positive-unlabelled learning.
It is organised as follows:
1. Preprocessing: create positive and unlabelled masks.
2. Sampling: sample batches with respect to positive ratio in the dataset.
3. Dataset and Dataloaders: Pytorch datastructures with respect to PU principles.
"""

import os
import pickle
from typing import List, Tuple

import numpy as np
import tifffile
import torch
from scipy import ndimage
from skimage import exposure
from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from ..utils import load_annotations, load_array
from .augment import get_augment_policy

# +------------------------------------------------------------------------------------------+ #
# |                    PREPROCESSING: POSITIVE & UNLABELLED MASKS CREATION                   | #
# +------------------------------------------------------------------------------------------+ #


def find_max_positive_ratio_slice(image: np.ndarray) -> np.ndarray:
    positive_ratios = image.sum(axis=(1, 2)) / np.prod(image.shape[1:])
    return image[np.argmax(positive_ratios)]


def remove_small_objects_by_area(mask: np.ndarray, area_min: int = 50) -> np.ndarray:
    new_mask = np.copy(mask).astype(int)
    labels = label(new_mask)
    areas = np.bincount(labels.flatten())
    for region in regionprops(labels):
        new_mask[region.slice] = areas[region.label]
    new_mask = new_mask >= area_min
    return new_mask


def get_num_pixels_per_particle(radius: int) -> int:
    size = 2 * radius + 1
    mask = np.zeros((size, size, size))
    coords = np.array(np.ogrid[:size, :size, :size], dtype=object)
    center = np.repeat(radius, 3)
    distance = np.linalg.norm(coords - center)
    mask[distance <= radius] = 1
    return mask.sum(dtype=int)


def make_U_mask(
    image: np.ndarray,
    gamma: float = 1,
    gain: float = 1,
    area_min: int = 50,
    dilation_iter: int = 10,
    return_all_steps: bool = False,
) -> np.ndarray:
    # STEP 1: Gamma correction
    step1 = exposure.adjust_gamma(image, gamma, gain)
    # STEP 2: GMM Fitting & Binarize via GMM means
    # GMM fits on one slice: whitest slice if no slice specified
    fit_slice = find_max_positive_ratio_slice(step1)
    classifier = GaussianMixture(n_components=2)
    classifier.fit(fit_slice.reshape((fit_slice.size, 1)))
    threshold = np.mean(classifier.means_)
    step2 = step1 > threshold
    # STEP 3: Remove Small Objects
    step3 = remove_small_objects_by_area(step2, area_min=area_min)
    # STEP 4: Erosion: make a little 3D cube and erode 1 time with it
    cube = ndimage.generate_binary_structure(rank=3, connectivity=3)
    step4 = ndimage.binary_erosion(step3, cube, iterations=1)
    # STEP 5: Dilation: then dilate n times with it
    step5 = ndimage.binary_dilation(
        step4, cube, iterations=dilation_iter, brute_force=True
    )
    return (step1, step2, step3, step4, step5) if return_all_steps else step5


def recenter_if_needed(
    dim: Tuple[int], shape: Tuple[int], z: int, y: int, x: int
) -> Tuple[int]:
    d, h, w = dim
    D, H, W = shape
    z_min, y_min, x_min = z - d // 2, y - h // 2, x - w // 2
    z_min, y_min, x_min = max(0, z_min), max(0, y_min), max(0, x_min)
    z_max, y_max, x_max = z_min + d, y_min + h, x_min + w
    z_max, y_max, x_max = min(D, z_max), min(H, y_max), min(W, x_max)
    z_min, y_min, x_min = z_max - d, y_max - h, x_max - w
    return x_min, x_max, y_min, y_max, z_min, z_max


def make_P_mask(shape: Tuple[int], centers: np.ndarray, radius: int = 10) -> np.ndarray:
    size = 2 * radius + 1
    dim = np.min([shape, [size, size, size]], axis=0)
    base_mask = np.zeros(dim)
    coords = np.array(np.ogrid[: dim[0], : dim[1], : dim[2]], dtype=object)
    center = dim // 2
    distance = np.linalg.norm(coords - center)
    base_mask[distance <= radius] = 1
    mask = np.zeros(shape)
    for center in centers:
        x_min, x_max, y_min, y_max, z_min, z_max = recenter_if_needed(
            dim, shape, *center
        )
        block = mask[z_min:z_max, y_min:y_max, x_min:x_max]
        mask[z_min:z_max, y_min:y_max, x_min:x_max] = np.logical_or(block, base_mask)
    return mask


def prepare_pu_data(
    rootdir: str,
    output_dir: str,
    csv_name: str,
    extension: str,
    radius: int,
    shuffle: bool,
    num_pos_samples: int = None,
    load_u_masks: bool = False,
    save_selected_particles: bool = False,
) -> Tuple[np.ndarray]:
    print(f"| Preparing data from {rootdir}")
    # 1. Load images
    print("| STEP [1/3]: Loading images and annotations ...")
    images_names = sorted(
        list(filter(lambda x: x.endswith(extension), os.listdir(rootdir)))
    )
    images_paths = sorted(list(map(lambda x: os.path.join(rootdir, x), images_names)))
    images = [load_array(path) for path in tqdm(images_paths)]
    # 2. Load and sample annotations
    annotations = load_annotations(os.path.join(rootdir, csv_name))
    annotations = annotations[:, [0, 2, 3, 4]]
    annotations[:, [1, 2, 3]] = annotations[:, [1, 2, 3]].astype(int)
    num = np.zeros(annotations.shape[0], dtype=int)
    for image_name in images_names:
        idx_image = annotations[:, 0] == image_name
        num[idx_image] = np.arange(idx_image.sum())
    annotations_range = np.arange(annotations.shape[0])
    if shuffle:
        np.random.shuffle(annotations_range)
    annotations = annotations[annotations_range[:num_pos_samples]]
    if save_selected_particles:
        with open(os.path.join(output_dir, "selected_particles.pickle"), "wb") as file:
            ann_to_save = np.concatenate(
                (num[annotations_range[:num_pos_samples], None], annotations), axis=1
            )
            pickle.dump(ann_to_save, file)
    # 3. Load or Make PU masks
    if load_u_masks:
        print("| STEP [2/3]: Loading U masks ...")
        U_masks_dir = os.path.join(rootdir, "U_masks")
        U_masks = [
            tifffile.imread(os.path.join(U_masks_dir, name))
            for name in tqdm(images_names)
        ]
    else:
        print("| STEP [2/3]: Making U masks ...")
        U_masks = [make_U_mask(image) for image in images]
    print("| STEP [3/3]: Making P masks ...")
    P_masks = [
        make_P_mask(images[i].shape, annotations[annotations[:, 0] == name, 1:], radius)
        for i, name in tqdm(enumerate(images_names), total=len(images_names))
    ]
    return images, U_masks, P_masks


# +------------------------------------------------------------------------------------------+ #
# |                            POSITIVE - UNLABELLED BATCH SAMPLING                          | #
# +------------------------------------------------------------------------------------------+ #


class ShuffledSampler(Sampler):
    def __init__(self, dataset: np.ndarray) -> None:
        self.dataset = dataset
        self.reset_and_shuffle()

    def reset_and_shuffle(self) -> None:
        self.current_index = 0
        np.random.shuffle(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __next__(self) -> Tuple[int, int]:
        if self.current_index >= len(self.dataset):
            self.reset_and_shuffle()
        sample = self.dataset[self.current_index]
        self.current_index += 1
        return sample

    def __iter__(self) -> Sampler:
        return self


class PUSampler(Sampler):
    def __init__(
        self,
        P_masks: List[np.ndarray] = [],
        U_masks: List[np.ndarray] = [],
        pi: float = 0.5,
        batch_size: int = 64,
        epoch_size: int = None,
    ) -> None:
        P = self.enumerate_positive_coordinates(P_masks)
        U = self.enumerate_positive_coordinates(U_masks)
        self.P = ShuffledSampler(P)
        self.U = ShuffledSampler(U)
        self.pi = pi
        self.batch_size = batch_size
        self.num_positives_per_batch = int(pi * batch_size)
        if epoch_size is not None:
            self.size = epoch_size * batch_size
        else:
            self.size = len(self.P) + len(self.U)
        self.__reset_state()

    @staticmethod
    def enumerate_positive_coordinates(masks: np.ndarray) -> np.ndarray:
        size = int(sum(mask.sum() for mask in masks))
        positive_coordinates = np.zeros(
            size, dtype=[("image", np.uint32), ("coord", np.uint32)]
        )
        index = 0
        for mask_index, mask in enumerate(masks):
            positive_indices = np.flatnonzero(mask)
            num_positive_indices = len(positive_indices)
            positive_coordinates["image"][index : index + num_positive_indices] = (
                mask_index
            )
            positive_coordinates["coord"][index : index + num_positive_indices] = (
                positive_indices
            )
            index += num_positive_indices
        return positive_coordinates

    def __reset_state(self) -> None:
        self.current_sampler = self.P
        self.current_num_positives = 0
        self.current_index = 0

    def __update_state(self) -> None:
        self.current_index += 1
        if self.current_num_positives < self.num_positives_per_batch:
            self.current_num_positives += 1
        elif self.current_num_positives == self.num_positives_per_batch:
            self.current_sampler = self.U

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def encode_indices_to_integer(mask_index: int, coord: int) -> np.int64:
        return mask_index * 2**32 + coord

    def __next__(self) -> int:
        self.__update_state()
        mask_index, coord = next(self.current_sampler)
        if self.current_index == self.batch_size:
            self.__reset_state()
        return self.encode_indices_to_integer(mask_index, coord)

    def __iter__(self) -> int:
        for _ in range(len(self)):
            yield next(self)


def estimate_pi(
    U_masks: List[np.ndarray], expected_num_particles_per_image: int, radius: int
) -> float:
    num_images = len(U_masks)
    num_pixels_per_particle = get_num_pixels_per_particle(radius)
    expected_P_size = (
        num_pixels_per_particle * expected_num_particles_per_image * num_images
    )
    U_size = int(sum(mask.sum() for mask in U_masks))
    estimated_pi = expected_P_size / U_size
    return estimated_pi


# +------------------------------------------------------------------------------------------+ #
# |                                   DATASETS & DATALOADERS                                 | #
# +------------------------------------------------------------------------------------------+ #


class Trainset(Dataset):
    def __init__(
        self,
        images: Tuple[np.ndarray],
        labels: Tuple[np.ndarray],
        patch_size: Tuple[int],
        dim: int,
        augment: float,
        training: bool,
    ) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.patch_size = patch_size
        self.dim = dim
        self.augment = get_augment_policy(patch_size, p=augment, dim=dim)
        self.training = training

    def __len__(self):
        return int(sum(label.sum() for label in self.labels))

    @staticmethod
    def decode_integer_to_indices(encoding_integer: int) -> Tuple[int]:
        image_index = encoding_integer // 2**32
        coord = encoding_integer - image_index * 2**32
        return image_index, coord

    def get_patch_slices(self, shape, z, y, x) -> Tuple[int]:
        d, h, w = self.patch_size
        D, H, W = shape
        z_min, y_min, x_min = z - d // 2, y - h // 2, x - w // 2
        z_min, y_min, x_min = max(0, z_min), max(0, y_min), max(0, x_min)
        z_max, y_max, x_max = z_min + d, y_min + h, x_min + w
        z_max, y_max, x_max = min(D, z_max), min(H, y_max), min(W, x_max)
        z_min, y_min, x_min = z_max - d, y_max - h, x_max - w
        return slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max)

    def get_patch(self, image_index: int, coord: int) -> np.ndarray:
        image = self.images[image_index]
        d, h, w = image.shape
        z = coord // (h * w)
        xy = coord % (h * w)
        y = xy // w
        x = xy % w
        patch = image[self.get_patch_slices(image.shape, z, y, x)]
        if self.dim == 2:
            patch = patch.sum(axis=0)
            mini, maxi = patch.min(), patch.max()
            if mini != maxi:
                patch = (patch - mini) / (maxi - mini)
            else:
                patch = patch - mini
        if self.training:
            patch = self.augment(image=patch)["image"]
        patch = (patch - patch.min()) / (patch.max() - patch.min())
        return patch

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        image_index, coord = self.decode_integer_to_indices(idx)
        patch = self.get_patch(image_index, coord)
        label = self.labels[image_index].ravel()[coord]
        return patch, label


def collate_fn(batch):
    """Shuffle samples and put them into tensors."""
    patches = np.array([sample[0] for sample in batch])
    targets = np.array([sample[1] for sample in batch])
    indices = np.arange(len(batch))
    np.random.shuffle(indices)
    patches = torch.from_numpy(patches[indices])
    targets = torch.from_numpy(targets[indices])
    inputs = torch.unsqueeze(patches, dim=1).float()
    targets = torch.unsqueeze(targets, dim=1).float()
    return inputs, targets


# +------------------------------------------------------------------------------------------+ #
# |                                       EXTERNAL CALL                                      | #
# +------------------------------------------------------------------------------------------+ #


def make_dataloaders(
    rootdir: str,
    output_dir: str,
    batch_size: int,
    num_workers: int,
    num_pos_samples: int,
    shuffle: bool,
    augment: float,
    epoch_size: int,
    dim: int,
    extension: str,
    radius: int,
    num_particles_per_image: int,
    patch_size: Tuple[int],
    load_u_masks: bool,
) -> Tuple[torch.utils.data.DataLoader]:
    # 1. Load train images and make train PU masks
    datadir, csv_name = os.path.join(rootdir, "train"), "train_coordinates.csv"
    args = [
        datadir,
        output_dir,
        csv_name,
        extension,
        radius,
        shuffle,
        num_pos_samples,
        load_u_masks,
        True,
    ]
    train_images, train_U_masks, train_P_masks = prepare_pu_data(*args)
    # 2. Load val images and make val PU masks
    datadir, csv_name = os.path.join(rootdir, "val"), "val_coordinates.csv"
    args = [
        datadir,
        output_dir,
        csv_name,
        extension,
        radius,
        False,
        None,
        load_u_masks,
        False,
    ]
    val_images, val_U_masks, val_P_masks = prepare_pu_data(*args)
    print(len(val_images), len(val_P_masks), len(val_U_masks))
    print(len(train_images), len(train_P_masks), len(train_U_masks))
    # 3. Make train and val sets
    train_set = Trainset(
        train_images, train_P_masks, patch_size, dim, augment, training=True
    )
    val_set = Trainset(
        val_images, val_P_masks, patch_size, dim, augment, training=False
    )
    # 4. Estimate pi
    pi = estimate_pi([*train_U_masks, *val_U_masks], num_particles_per_image, radius)
    # 5. Make train and val samplers
    kwargs = {"pi": pi, "batch_size": batch_size, "epoch_size": epoch_size}
    train_sampler = PUSampler(train_P_masks, train_U_masks, **kwargs)
    val_sampler = PUSampler(val_P_masks, val_U_masks, **kwargs)
    # 6. Make train and val loaders
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(train_set, sampler=train_sampler, **kwargs)
    val_loader = DataLoader(val_set, sampler=val_sampler, **kwargs)
    return train_loader, val_loader
