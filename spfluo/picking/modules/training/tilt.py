"""The pretraining submodule aims at generating datasets, and preparing data for training
(crop + split) regardless of the task. The other two submodules (training and postraining)
however  are specific to the picking task. That is why everything related to tilt
(data loading, training, prediction) is contained separately here within a single file.
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils import load_annotations, load_array, summary
from .augment import get_augment_policy
from .core import Accuracy, Data, Logger, Losses, fit, make_model
from .prepare_fs_data import get_patch_paths
from .unsupervised_pretraining import unsupervised_pretraining

# +------------------------------------------------------------------------------------------+ #
# |                                          DATASET                                         | #
# +------------------------------------------------------------------------------------------+ #


class Trainset(Dataset):
    def __init__(
        self,
        paths: List[str],
        labels: np.ndarray,
        patch_size: Tuple[int],
        extension: str,
        augment: float,
        training: bool,
        size: int = None,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.extension = extension
        self.augment = get_augment_policy(patch_size, p=augment, dim=2)
        self.training = training
        self.size = size
        self.unsupervised = self.labels is None

    def get_target(self, path: str) -> int:
        filename = os.path.basename(path)
        if self.extension == "npz":
            y = filename.split("_")
            image_name, patch_index = "_".join(y[:-1]), y[-1]
        else:
            temp = filename.split("_")
            image_name, patch_index = "_".join(temp[:-1]), temp[-1]
        patch_index = int(patch_index.split(".")[0])
        image_name = f"{image_name}.{self.extension}"
        target = self.labels[self.labels[:, 0] == image_name][patch_index][1]
        return torch.as_tensor(target)

    def get_image(self, path: str):
        image = load_array(path)
        image = image.sum(axis=0)  # sum over z axis
        image = (image - image.min()) / (image.max() - image.min())
        if self.training:
            image = self.augment(image=image)["image"]
        image = (image - image.min()) / (image.max() - image.min())
        image = np.expand_dims(image, axis=0)  # add channel dim at the beginning
        return torch.from_numpy(image.astype(np.float32))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if index >= len(self.paths):
            np.random.shuffle(self.paths)
        index = index % len(self.paths)
        target = self.get_target(self.paths[index]) if not self.unsupervised else 0
        return self.get_image(self.paths[index]), target

    def __len__(self) -> int:
        return self.size if self.size is not None else len(self.paths)


def get_patch_size(rootdir: str) -> Tuple[int]:
    train_paths = get_patch_paths(os.path.join(rootdir, "train", "cropped"))
    return load_array(train_paths[0]).shape


# +------------------------------------------------------------------------------------------+ #
# |                                    TRAINING STRUCTURES                                   | #
# +------------------------------------------------------------------------------------------+ #


def make_data(
    rootdir: str,
    output_dir: str,
    extension: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    augment: float,
    size: int,
    patch_size: Tuple[int],
    num_pos_samples: int,
    unsupervised: bool,
    selected_particles: str,
    crop_dir: str,
) -> Tuple[DataLoader]:
    # labels
    if unsupervised:
        train_views, val_views = None, None
    else:
        if extension == "npz":
            views = load_annotations(os.path.join(rootdir, "views.csv"))
            train_views = val_views = views
        else:
            train_views = load_annotations(
                os.path.join(rootdir, "train", "train_views.csv")
            )
            val_views = load_annotations(os.path.join(rootdir, "val", "val_views.csv"))

    if crop_dir is None:
        datadir = os.path.join(rootdir, "train", "cropped")
    else:
        datadir = crop_dir

    # train data
    if selected_particles is None:
        train_paths = get_patch_paths(datadir, num_pos_samples, shuffle)
        train_paths = sorted(list(filter(lambda x: "positive" in x, train_paths)))
    else:
        with open(selected_particles, "rb") as f:
            picked_particles = pickle.load(f)
        width = int(
            np.ceil(np.log10(len(os.listdir(os.path.join(datadir, "positive")))))
        )
        train_paths = list(
            map(
                lambda x: os.path.join(
                    datadir,
                    "positive",
                    x[1].replace(
                        "." + extension, "_" + str(x[0]).zfill(width) + ".npz"
                    ),
                ),
                picked_particles,
            )
        )
    with open(os.path.join(output_dir, "selected_particles.pickle"), "wb") as file:
        pickle.dump(train_paths, file)
    args = [train_paths, train_views, patch_size, extension, augment]
    # print(train_paths)
    train_set = Trainset(*args, training=True, size=size)
    # val data
    if crop_dir is None:
        datadir = os.path.join(rootdir, "val", "cropped")
    else:
        datadir = crop_dir
    val_paths = get_patch_paths(datadir)  # num_pos_samples=None, shuffle=False
    val_paths = sorted(list(filter(lambda x: "positive" in x, val_paths)))
    val_set = Trainset(
        val_paths, val_views, patch_size, extension, augment=0, training=False
    )
    # loaders
    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers}
    train_loader = DataLoader(train_set, **loader_kwargs, shuffle=True)
    val_loader = DataLoader(val_set, **loader_kwargs, shuffle=False)
    return Data(train_loader, val_loader)


def make_logger(config: Dict) -> Logger:
    # Names order MUST match MultiTaskLoss outputs order.
    loss_names = ["classification"]
    if config["ae"]:
        loss_names.append("reconstruction")
    loss_names.append("total")
    losses = Losses(loss_names)
    accuracy = Accuracy()
    return Logger(losses, accuracy, config)


# +------------------------------------------------------------------------------------------+ #
# |                                       EXTERNAL CALL                                      | #
# +------------------------------------------------------------------------------------------+ #


def train(
    rootdir: str,
    batch_size: int,
    num_workers: int,
    output_dir: str,
    extension: str,
    selected_particles: str,
    crop_dir: str,
    num_epochs: int,
    lr: float,
    network: str,
    width_factor: float,
    depth_factor: float,
    swa: bool,
    swa_lr: float,
    patch_size: Tuple[int],
    num_pos_samples: int,
    shuffle: bool,
    augment: float,
    size: int = None,
    # autoencoder params
    ae: bool = False,
    eta: float = 0.0,
    learn_eta: bool = False,
    num_features: Tuple[int] = None,
    hidden_dim: int = None,
    # unsupervised pretraining
    pretrain: bool = False,
) -> None:
    print("\nPreparing training pipeline ...\n")
    config = locals()
    os.makedirs(output_dir, exist_ok=True)
    summary_output = os.path.join(output_dir, "tilt_train_config.txt")
    summary(config, title="TILT TRAINING CONFIG", output=summary_output)
    num_classes, dim, pi = 3, 2, None
    data_params = [
        rootdir,
        output_dir,
        extension,
        batch_size,
        num_workers,
        shuffle,
        augment,
        size,
        patch_size,
    ]
    train_params = [num_classes, dim, lr, patch_size, num_epochs, pi]
    model_params = [network, width_factor, depth_factor, swa, swa_lr]
    ae_params = [ae, eta, learn_eta, num_features, hidden_dim]
    if pretrain:
        # autoencoder reconstruction task for unsupervised pretraining of the encoder.
        data = make_data(
            *data_params, None, True, None, crop_dir
        )  # num_pos_samples=None <=> use all samples
        model = make_model(
            *train_params, *model_params, *[True, *ae_params[1:]]
        )  # ae = True
        model.train_criterion = model.val_criterion = torch.nn.MSELoss()
        model = unsupervised_pretraining(data, model, num_epochs, output_dir)
        pretrained_encoder_state = model.network.state_dict()
    data = make_data(*data_params, num_pos_samples, False, selected_particles, None)
    model = make_model(*train_params, *model_params, *ae_params)
    if pretrain:
        model.network.load_state_dict(pretrained_encoder_state)
    logger = make_logger(config)
    fit(data, model, logger)
