"""Basically nothing more than classic Pytorch dataloaders creation. This only external call
should be to the 'get_dataloaders' function. Other functions defined here should only be used
here internally by the 'get_dataloaders' function.

The data folder must has the following structure:

rootdir
|
|__train
|    |___ cropped
|            |___ positive
|            |___ negative
|__ val
     |___ cropped
             |___ positive
             |___ negative
"""

import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils import load_array
from .augment import get_augment_policy

# +------------------------------------------------------------------------------------------+ #
# |                                          DATASET                                         | #
# +------------------------------------------------------------------------------------------+ #


def get_patch_paths(
    rootdir: str, num_pos_samples: int = None, shuffle: bool = False
) -> List[str]:
    all_paths = []
    for subdir in ["positive", "negative"]:
        datadir = os.path.join(rootdir, subdir)
        paths = sorted(
            list(map(lambda x: os.path.join(datadir, x), os.listdir(datadir)))
        )
        if shuffle:
            np.random.shuffle(paths)
        all_paths.extend(paths[:num_pos_samples])
    if shuffle:
        np.random.shuffle(all_paths)
    return all_paths


class Trainset(Dataset):
    def __init__(
        self,
        paths: List[str],
        dim: int,
        augment: float,
        size: int = None,
        training: bool = True,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.dim = dim
        patch_size = load_array(self.paths[0]).shape
        self.augment = get_augment_policy(patch_size, p=augment, dim=dim)
        self.training = training
        self.size = size

    @staticmethod
    def get_target(path: str) -> int:
        target = 1 if "positive" in path else 0
        return torch.as_tensor(target).unsqueeze(dim=-1).float()

    def get_image(self, path: str):
        image = load_array(path)
        if self.dim == 2:
            image = image.sum(axis=0)
            image = (image - image.min()) / (image.max() - image.min())
        if self.training:
            image = self.augment(image=image)["image"]
        image = (image - image.min()) / (image.max() - image.min())
        image = np.expand_dims(image, axis=0)  # add channel dim at the beginning
        return torch.from_numpy(image.astype(np.float32))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        if index >= len(self.paths):
            np.random.shuffle(self.paths)
        index = index % len(self.paths)
        return self.get_image(self.paths[index]), self.get_target(self.paths[index])

    def __len__(self) -> int:
        return self.size if self.size is not None else len(self.paths)


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
) -> Tuple[DataLoader]:
    # train data
    datadir = os.path.join(rootdir, "train", "cropped")
    train_paths = get_patch_paths(datadir, num_pos_samples, shuffle)
    with open(os.path.join(output_dir, "selected_particles.pickle"), "wb") as file:
        pickle.dump(train_paths, file)
    size = batch_size * epoch_size if epoch_size is not None else None
    train_set = Trainset(train_paths, dim, augment, size, training=True)
    # val data
    datadir = os.path.join(rootdir, "val", "cropped")
    val_paths = get_patch_paths(datadir)  # num_pos_samples=None, shuffle=False
    val_set = Trainset(val_paths, dim, augment, training=False)
    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers}
    train_loader = DataLoader(train_set, **loader_kwargs, shuffle=True)
    val_loader = DataLoader(val_set, **loader_kwargs, shuffle=False)
    return train_loader, val_loader
