"""A training pipeline which aims at being simple and modular. It is organized as follows:

1. Define three structures for training: Data, Model and Logger.
    - Data: a wrapper around train and val dataloaders.
    - Model: a wrapper containing:
        * a device (GPU or CPU basically)
        * a network
        * a decoder (optional)
        * an optimizer
        * a learning rate scheduler
        * a criterion (wrapper around an arbitrary number of criteria)
    - Logger: based on DataTracker, a convenient workaround to handle an arbitrary number of metrics
              and losses during training. It handles printing, tensorboard logging, and training
              state saving.

2. Define a Model factory function make_model handling:
    - training mode (PU or FS)
    - task (picking or tilt)
    - training config (num epochs, lr, backbone, swa, dim)
    - optionaly, reconstruction branch (autoencoder regularization)

3. Define functions to perform training and validation loop with logging.

In order to be used, one must simply instanciate a Data and Logger, call the make_model function,
and finally call the fit function accordingly.
That's what picking.py and tilt.py do.
"""

import functools
import os
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from tqdm import tqdm

from .losses import BinomialGE, MultiTaskLoss
from .networks import Decoder, EfficientNet

# +------------------------------------------------------------------------------------------+ #
# |                                    TRAINING COMPONENTS                                   | #
# +------------------------------------------------------------------------------------------+ #


# ____________________________________________ DATA __________________________________________ #


@dataclass
class Data:
    """Nothing more than a wrapping structure handy for training functions."""

    train_loader: DataLoader
    val_loader: DataLoader


# ___________________________________________ MODEL __________________________________________ #


@dataclass
class Model:
    """Nothing more than a wrapping structure handy for training functions."""

    device: torch.device
    network: torch.nn.Module
    decoder: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    train_criterion: MultiTaskLoss
    val_criterion: MultiTaskLoss


# ________________________________________ DATA TRACKER ______________________________________ #


class DataTracker:
    def __init__(self, names: List[str], init_best: float = 0.0) -> None:
        self.names = names
        self.total = 0
        self.current = {}
        self.best = self.init_best(init_best)
        self.reset()

    def init_best(self, value: float) -> None:
        best = {}
        for name in self.names:
            best[name] = value
        return best

    def step(self, *args, **kwargs) -> None:
        """Should update current and total."""
        pass

    def reduce(self) -> None:
        for name in self.names:
            self.current[name] /= self.total

    def reset(self) -> None:
        for name in self.names:
            self.current[name] = 0
        self.total = 0

    def update_best(self) -> None:
        for name in self.names:
            self.best[name] = self.current[name]


# _________________________________________ ACCURACY _________________________________________ #


class Accuracy(DataTracker):
    def __init__(self) -> None:
        super().__init__(["accuracy"], init_best=0)

    def step(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.current["accuracy"] += outputs.argmax(dim=-1).eq(targets).sum().item()
        self.total += len(targets)


# __________________________________________ F1 SCORE ________________________________________ #


class F1Score(DataTracker):
    def __init__(self) -> None:
        super().__init__(["precision", "recall", "f1-score"], init_best=0)
        self.reset()

    def step(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = outputs.detach().sigmoid().round()
        target_true = targets == 1
        predicted_true = predictions == 1
        self.correct_true += (
            (targets[target_true] == predictions[target_true]).sum().item()
        )
        self.target_true += target_true.sum().item()
        self.predicted_true += predicted_true.sum().item()

    def reduce(self) -> None:
        target_true_str = f"(target true: {self.target_true}, "
        predicted_true_str = f"predicted true: {self.predicted_true}, "
        correct_true_str = f"correct true: {self.correct_true})"
        print(target_true_str + predicted_true_str + correct_true_str)
        precision = (
            self.correct_true / self.predicted_true if self.predicted_true > 0 else 0
        )
        recall = self.correct_true / self.target_true if self.target_true > 0 else 0
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        self.current["precision"] = precision
        self.current["recall"] = recall
        self.current["f1-score"] = f1_score

    def reset(self) -> None:
        self.correct_true = 0
        self.target_true = 0
        self.predicted_true = 0


# ___________________________________________ LOSSES _________________________________________ #


class Losses(DataTracker):
    def __init__(self, names: List[str]) -> None:
        super().__init__(names, init_best=np.inf)

    def step(self, losses: List[Tensor], total_loss: Tensor, length: int) -> None:
        for i, name in enumerate(self.names[:-1]):  # exclude total
            self.current[name] += losses[i]
        self.current["total"] += total_loss
        self.total += length


# ___________________________________________ LOGGER _________________________________________ #


class SummaryWriter(SummaryWriter):
    """When logging hparams in addition to scalars, tensorboard will create a subfolder.
    This behavior allows to store multiple hparams config within the same run folder.
    However, it is not what we want here. This small workaround prevents that.
    From: https://github.com/pytorch/pytorch/issues/32651
    """

    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if not isinstance(hparam_dict, dict) or not isinstance(metric_dict, dict):
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)
        logdir = self._get_file_writer().get_logdir()
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


class Logger:
    def __init__(self, losses: Losses, metrics: DataTracker, config: Dict) -> None:
        self.losses = losses
        self.metrics = metrics
        self.config = config
        self.epoch = 0
        self.epochs = config["num_epochs"]
        self.log_dir = config["output_dir"]
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.prefix_length = self.set_prefix_length()

    def set_prefix_length(self) -> None:
        max_losses_length = max([len(name) for name in self.losses.names])
        max_metrics_length = max([len(name) for name in self.metrics.names])
        return max(max_losses_length, max_metrics_length)

    def step(
        self, losses: List[Tensor], total_loss: Tensor, outputs: Tensor, targets: Tensor
    ):
        self.losses.step(losses, total_loss, len(targets))
        self.metrics.step(outputs, targets)

    def reduce(self) -> None:
        self.losses.reduce()
        self.metrics.reduce()

    def reset(self) -> None:
        self.losses.reset()
        self.metrics.reset()

    def to_tensorboard(self, training: bool) -> None:
        stage = "Train" if training else "Validation"
        for name, loss in self.losses.current.items():
            self.writer.add_scalar(
                f"{name.capitalize()} Loss/{stage}", loss, self.epoch
            )
        for name, metric in self.metrics.current.items():
            self.writer.add_scalar(f"{name.capitalize()}/{stage}", metric, self.epoch)

    def hparams_and_metrics_to_tensorboard(self) -> None:
        metrics_dict = {f"hparams/{k}": v for k, v in self.metrics.best.items()}
        # tensorboard accepts tensor but not tuple or list ...
        for k, v in self.config.items():
            if isinstance(v, Iterable) and not isinstance(v, str):
                self.config[k] = torch.as_tensor(v)
        for k, v in metrics_dict.items():
            if isinstance(v, Iterable):
                metrics_dict[k] = torch.as_tensor(v)
        self.writer.add_hparams(self.config, metrics_dict)

    def save_training_state(self, model: Model) -> None:
        print("Saving new best model.")
        state = {
            "losses": self.losses.best,
            "metrics": self.metrics.best,
            "network_state_dict": model.network.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict(),
        }
        path = os.path.join(self.log_dir, "checkpoint.pt")
        torch.save(state, path)

    def print_dict(self, dict: Dict, prefix: str = None) -> None:
        for k, v in dict.items():
            print(f"{prefix} {k}{(self.prefix_length - len(k)) * '.'}: {v:.4f}")

    def on_full_epoch_start(self) -> None:
        print(f"\nEPOCH [{self.epoch + 1}/{self.epochs}]")

    def on_full_epoch_end(self, model: Model) -> None:
        self.epoch += 1
        lr = model.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("Learning Rate", lr, self.epoch)
        eta = model.train_criterion.eta
        for i, e in enumerate(eta):
            self.writer.add_scalar(f"Loss Weights/eta_{i}", e, self.epoch)
        self.print_dict(self.losses.current, prefix="Validation")
        self.print_dict(self.metrics.current, prefix="Validation")
        if self.losses.current["total"] <= self.losses.best["total"]:
            self.losses.update_best()
            self.metrics.update_best()
            self.save_training_state(model)

    def on_fit_end(self, model: Model) -> None:
        print(40 * "-")
        self.print_dict(self.losses.best, prefix="Best validation")
        self.print_dict(self.metrics.best, prefix="Best validation")
        print(f"\nNetwork num params: {model.network.num_params}")
        if model.decoder is not None:
            print(f"Decoder num params: {model.decoder.num_params}")
        print(40 * "-")
        self.hparams_and_metrics_to_tensorboard()


# +------------------------------------------------------------------------------------------+ #
# |                                       MODEL FACTORY                                      | #
# +------------------------------------------------------------------------------------------+ #


def make_model(
    # train params
    num_classes: str,
    dim: int,
    lr: float,
    patch_size: Tuple[int],
    num_epochs: int,
    pi: float,
    # model params
    network: str,
    width_factor: float,
    depth_factor: float,
    swa: bool,
    swa_lr: float,
    # autoencoder params
    ae: bool = False,
    eta: float = 0.0,
    learn_eta: bool = False,
    num_features: Tuple[int] = None,
    hidden_dim: int = None,
) -> Model:
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. Network
    kwargs = dict(
        dim=dim,
        num_classes=num_classes,
        width_factor=width_factor,
        depth_factor=depth_factor,
    )
    network = EfficientNet.from_name(network, **kwargs).to(device)
    params = [{"params": network.parameters()}]
    # 3. Classification loss
    # train and val losses are separated because in PU we don't use the same one
    # in train (GE Binomial) and in val (Binary Cross Entropy).
    if num_classes == 3:  # tilt
        train_criteria = [torch.nn.CrossEntropyLoss()]
        val_criteria = [torch.nn.CrossEntropyLoss()]
    else:  # picking
        # pi is None <=> FS or PU
        train_criteria = [
            BinomialGE(pi=pi) if pi is not None else torch.nn.BCEWithLogitsLoss()
        ]
        val_criteria = [torch.nn.BCEWithLogitsLoss()]
    # 4. Decoder and Reconstruction loss (optional)
    decoder = None
    if ae:
        decoder_kwargs = {
            "latent_dim": network.latent_dim,
            "hidden_dim": hidden_dim,
            "num_features": num_features,
            "dim": dim,
        }
        decoder = Decoder(out_size=patch_size, **decoder_kwargs).to(device)
        params.append({"params": decoder.parameters()})
        train_criteria.append(torch.nn.MSELoss())
        val_criteria.append(torch.nn.MSELoss())
    # 5. Merged loss
    multi_task_loss_kwargs = {
        "eta": len(train_criteria) * [eta],
        "learn_eta": learn_eta,
        "device": device,
    }
    train_criterion = MultiTaskLoss(train_criteria, **multi_task_loss_kwargs)
    if learn_eta:
        params.append({"params": train_criterion.parameters()})
    val_criterion = MultiTaskLoss(val_criteria, device=device)
    # 6. Optimizer
    adam_params = {
        "lr": lr,
        "betas": (0.9, 0.999),
        "weight_decay": 0.01,
        "amsgrad": False,
    }
    optimizer = torch.optim.AdamW(params, **adam_params)
    # sgd_params = {'lr': lr, 'momentum': 0.9, 'weight_decay': 5e-4}
    # optimizer = torch.optim.SGD(params, **sgd_params)
    # 7. Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # scheduler = None
    model = Model(
        device, network, decoder, optimizer, scheduler, train_criterion, val_criterion
    )
    # 8. Stochastic Weights Averaging (optional)
    model.swa = swa
    if model.swa:

        def ema_avg(averaged_model_parameter, model_parameter, num_averaged):
            return 0.1 * averaged_model_parameter + 0.9 * model_parameter

        ema_model = torch.optim.swa_utils.AveragedModel(model.network, avg_fn=ema_avg)
        model.swa_network = ema_model
        model.swa_scheduler = torch.optim.swa_utils.SWALR(
            model.optimizer, swa_lr=swa_lr
        )
        model.swa_start = num_epochs // 2
    return model


# +------------------------------------------------------------------------------------------+ #
# |                               TRAIN/VAL AUXILIARY FUNCTIONS                              | #
# +------------------------------------------------------------------------------------------+ #


def early_stopping(patience: int = 10):
    """Outer wrapper because the decorator has argument(s)."""

    def wrapper(epoch_function):
        epoch_function.current_run = 0
        """ The actual decorator. """

        @functools.wraps(epoch_function)
        def inner(*args, **kwargs):
            model, logger = epoch_function(*args, **kwargs)
            if logger.losses.current["total"] < logger.losses.best["total"]:
                epoch_function.current_run = 0  # reset patience
            else:
                epoch_function.current_run += 1
            if epoch_function.current_run == patience:
                print(f"Early stopping triggered (patience {patience}).")
                # will be used to detect early stopping in fit function
                logger.losses.current["total"] = None
            return model, logger

        return inner

    return wrapper


def timer(func):
    """A simple decorator to measure the execution time of a given function."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = int(toc - tic)
        s = elapsed_time % 60
        m = (elapsed_time // 60) % 60
        h = elapsed_time // 3600
        formatted_elapsed_time = "{:d}h {:02d}m {:02d}s".format(h, m, s)
        print(f"Elapsed time: {formatted_elapsed_time}.")
        return value

    return wrapper_timer


def complete_swa(loader: DataLoader, model: Model, log_dir: str) -> None:
    print("Taking care of SWA batch norm statistics.")
    torch.optim.swa_utils.update_bn(loader, model.swa_network, model.device)
    print("Saving SWA weights.")
    # Make SWA modules names equal to original network ones.
    swa_state = model.swa_network.state_dict()
    #   - delete SWA specific key
    del swa_state["n_averaged"]
    #   - remove 'module.' prefix
    modules = [(".".join(k.split(".")[1:]), v) for k, v in swa_state.items()]
    #   - wrap results
    new_swa_state = OrderedDict(modules)
    # Save
    state = dict(network_state_dict=new_swa_state)
    torch.save(state, os.path.join(log_dir, "checkpoint_swa.pt"))


# +------------------------------------------------------------------------------------------+ #
# |                                  TRAIN/VAL MAIN FUNCTIONS                                | #
# +------------------------------------------------------------------------------------------+ #


def step(x: Tensor, y: Tensor, model: Model, logger: Logger, training: bool) -> None:
    x, y = x.to(model.device), y.to(model.device)
    z = model.network.extract_features(x)
    y_hat = model.network.head(z)
    outputs, targets = [y_hat], [y]
    if model.decoder is not None:
        model.val_criterion.eta = (
            model.train_criterion.eta
        )  # in case eta is being learnt
        x_hat = model.decoder(z)
        outputs.append(x_hat)
        targets.append(x)
    criterion = model.train_criterion if training else model.val_criterion
    losses, total_loss = criterion(outputs, targets)
    if training:
        model.optimizer.zero_grad()
        total_loss.backward()
        model.optimizer.step()
    losses = [loss.detach().item() for loss in losses]
    logger.step(losses, total_loss.detach().item(), y_hat.detach(), y)


def half_epoch(
    dataloader: DataLoader, model: Model, logger: Logger, training=True
) -> Model:
    model.network.train() if training else model.network.eval()
    torch.set_grad_enabled(training)
    for inputs, targets in tqdm(dataloader):
        step(inputs, targets, model, logger, training)
    logger.reduce()
    logger.to_tensorboard(training)
    return model


# @early_stopping(patience=10)
def full_epoch(data: Data, model: Model, logger: Logger) -> Tuple[Model, Logger]:
    model = half_epoch(data.train_loader, model, logger, training=True)
    logger.reset()
    if len(data.val_loader.dataset) > 0:
        model = half_epoch(data.val_loader, model, logger, training=False)
    else:
        print("Val data empty !")
    if model.swa and logger.epoch > model.swa_start:
        model.swa_network.update_parameters(model.network)
        model.swa_scheduler.step()
    elif model.scheduler is not None:
        model.scheduler.step()
    return model, logger


@timer
def fit(data: Data, model: Model, logger: Logger) -> None:
    for _ in range(logger.epochs):
        logger.on_full_epoch_start()
        model, logger = full_epoch(data, model, logger)
        if logger.losses.current["total"] is None:  # Early stopping
            break
        logger.on_full_epoch_end(model)
        logger.reset()
    logger.on_fit_end(model)
    if model.swa:
        complete_swa(data.train_loader, model, logger.log_dir)
