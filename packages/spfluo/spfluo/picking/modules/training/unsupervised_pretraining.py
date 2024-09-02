"""Almost a copy-paste from modules.training.core but adapted for reconstruction only."""

import os
import pickle

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .core import Data, Model


def step(x: Tensor, model: Model, training: bool) -> float:
    x = x.to(model.device)
    x_hat = model.decoder(model.network.extract_features(x))
    criterion = model.train_criterion if training else model.val_criterion
    loss = criterion(x_hat, x)
    if training:
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
    return loss.detach().item()


def half_epoch(dataloader: DataLoader, model: Model, training=True) -> Model:
    model.network.train() if training else model.network.eval()
    model.decoder.train() if training else model.decoder.eval()
    torch.set_grad_enabled(training)
    loss, length = 0, 0
    for inputs, _ in dataloader:
        loss += step(inputs, model, training)
        length += len(inputs)
    loss /= length
    return model, loss


def full_epoch(data: Data, model: Model) -> Model:
    model, train_loss = half_epoch(data.train_loader, model, training=True)
    model, val_loss = half_epoch(data.val_loader, model, training=False)
    if model.scheduler is not None:
        model.scheduler.step()
    return model, train_loss, val_loss


def unsupervised_pretraining(
    data: Data, model: Model, num_epochs: int, output_dir: str
) -> Model:
    train_losses, val_losses = [], []
    with tqdm(total=num_epochs) as pbar:
        for i in range(num_epochs):
            pbar.set_description(f"EPOCH [{i+1}/{num_epochs}]")
            model, train_loss, val_loss = full_epoch(data, model)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            pbar.set_postfix({"train loss": train_loss, "val_loss": val_loss})
            pbar.update()
    with open(
        os.path.join(output_dir, "unsupervised_pretraining_train_losses.pickle"), "wb"
    ) as f:
        pickle.dump(train_losses, f)
    with open(
        os.path.join(output_dir, "unsupervised_pretraining_val_losses.pickle"), "wb"
    ) as f:
        pickle.dump(val_losses, f)
    return model
