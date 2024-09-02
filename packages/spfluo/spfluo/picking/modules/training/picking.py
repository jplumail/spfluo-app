from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

from ..utils import summary
from .core import Data, F1Score, Logger, Losses, fit, make_model
from .prepare_fs_data import make_dataloaders as make_fs_dataloaders
from .prepare_pu_data import make_dataloaders as make_pu_dataloaders


def make_data(
    # base params
    mode: str,
    rootdir: str,
    output_dir: str,
    batch_size: int,
    num_workers: int,
    num_pos_samples: int,
    augment: bool,
    shuffle: bool,
    extension: str,
    dim: int,
    epoch_size: int = None,
    # pu optional params
    radius: int = None,
    num_particles_per_image: int = None,
    patch_size: Tuple[int] = None,
    load_u_masks: bool = False,
) -> Data:
    args = [
        rootdir,
        output_dir,
        batch_size,
        num_workers,
        num_pos_samples,
        shuffle,
        augment,
        epoch_size,
        dim,
    ]
    maker = make_fs_dataloaders
    if mode == "pu":
        args += [extension, radius, num_particles_per_image, patch_size, load_u_masks]
        maker = make_pu_dataloaders
    train_loader, val_loader = maker(*args)
    return Data(train_loader, val_loader)


def make_logger(config: Dict) -> Logger:
    # Names order MUST match MultiTaskLoss outputs order.
    loss_names = ["classification"]
    if config["ae"]:
        loss_names.append("reconstruction")
    loss_names.append("total")
    losses = Losses(loss_names)
    f1_score = F1Score()
    return Logger(losses, f1_score, config)


# +------------------------------------------------------------------------------------------+ #
# |                                       EXTERNAL CALL                                      | #
# +------------------------------------------------------------------------------------------+ #


def train(
    # base params
    mode: str,
    rootdir: str,
    batch_size: int,
    num_workers: int,
    output_dir: str,
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
    augment: bool,
    dim: int,
    extension: str,
    epoch_size: int = None,
    downscale: float = 1.0,
    # pu optional params
    radius: int | None = None,
    num_particles_per_image: int = None,
    load_u_masks: bool = False,
    # autoencoder params
    ae: bool = False,
    eta: float = 0.0,
    learn_eta: bool = False,
    num_features: Tuple[int] = None,
    hidden_dim: int = None,
) -> None:
    print("\nPreparing training pipeline ...\n")
    config = locals()
    os.makedirs(output_dir, exist_ok=True)
    summary_output = os.path.join(output_dir, "train_config.txt")
    summary(config, title="TRAINING CONFIG", output=summary_output)
    patch_size = np.rint(np.array(patch_size) / downscale).astype(int)
    radius = round(radius / downscale)
    data = make_data(
        mode,
        rootdir,
        output_dir,
        batch_size,
        num_workers,
        num_pos_samples,
        augment,
        shuffle,
        extension,
        dim,
        epoch_size,
        radius,
        num_particles_per_image,
        patch_size,
        load_u_masks,
    )
    num_classes = 1
    pi = data.train_loader.sampler.pi if mode == "pu" else None
    train_params = [num_classes, dim, lr, patch_size, num_epochs, pi]
    model_params = [network, width_factor, depth_factor, swa, swa_lr]
    ae_params = [ae, eta, learn_eta, num_features, hidden_dim]
    model = make_model(*train_params, *model_params, *ae_params)
    logger = make_logger(config)
    fit(data, model, logger)
