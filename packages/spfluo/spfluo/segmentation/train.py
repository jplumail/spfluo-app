import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from torch.optim import Adam
from torch.utils.data import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from .data import (
    Generator_variableSize,
    GeneratorDataset,
    HardCopyDataset,
    MultiEpochsDataLoader,
    infos_to_dataframe,
    unbatch_dict,
)
from .losses import iaawnLossDenis
from .unet_3d import UNet3D

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cudnn.benchmark = True
    num_epochs = 20
    lr = 1e-5
    alpha = 1
    beta = 1e-2
    n_samples_per_epoch = 200
    batch_size = 4
    n_samples_validation_sets = 50
    n_samples_test_sets = 1000

    n_training = 60
    exp_name = "exptest3-reg4"

    # Fix randomness
    torch.random.manual_seed(0)

    print("Init train loader")
    train_loader_a = MultiEpochsDataLoader(
        GeneratorDataset(Generator_variableSize(600), n_samples_per_epoch),
        batch_size=batch_size,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader_a2 = MultiEpochsDataLoader(
        GeneratorDataset(Generator_variableSize(100, 1000), n_samples_per_epoch),
        batch_size=batch_size,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader_b = MultiEpochsDataLoader(
        GeneratorDataset(Generator_variableSize(150), n_samples_per_epoch),
        batch_size=batch_size,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
    )

    print("Init valid loaders")
    print("A...")
    valid_loader_a = torch.utils.data.DataLoader(
        HardCopyDataset(
            GeneratorDataset(Generator_variableSize(600), n_samples_validation_sets),
            num_workers=32,
        ),
        batch_size=8,
        pin_memory=True,
    )
    print("B...")
    valid_loader_b = torch.utils.data.DataLoader(
        HardCopyDataset(
            GeneratorDataset(Generator_variableSize(150), n_samples_validation_sets),
            num_workers=32,
        ),
        batch_size=8,
        pin_memory=True,
    )
    print("A extended...")
    valid_loader_a2 = torch.utils.data.DataLoader(
        HardCopyDataset(
            GeneratorDataset(
                Generator_variableSize(100, 1000), n_samples_validation_sets
            ),
            num_workers=32,
        ),
        batch_size=8,
        pin_memory=True,
    )

    print("Init testing loader...")
    testing_loader = torch.utils.data.DataLoader(
        HardCopyDataset(
            GeneratorDataset(Generator_variableSize(100, 1000), n_samples_test_sets),
            num_workers=32,
        ),
        batch_size=8,
        pin_memory=True,
        num_workers=8,
    )

    loss = iaawnLossDenis(normalization="sigmoid", alpha=alpha, beta=beta)
    metric = iaawnLossDenis(normalization="sigmoid", alpha=1, beta=1e-2)

    for name_train, train_loader, valid_loader_early_stopping in zip(
        ["A", "C", "B"],
        [train_loader_a, train_loader_a2, train_loader_b],
        [valid_loader_a, valid_loader_a2, valid_loader_b],
    ):
        # for name_train, train_loader in zip(["A"], [train_loader_a]):
        print(f"Start {name_train} training")
        for i in range(n_training):
            print("Init Unet...")
            unet = UNet3D(1, 1).to("cuda")
            print("Init optimizer...")
            optimizer = Adam(unet.parameters(), lr=lr)

            # Logs
            writer = SummaryWriter(comment=f"_{exp_name}-{name_train}-{i}")

            metrics = {"val-A": [], "val-B": [], "val-C": [], "train": []}
            best_loss = np.inf
            for j in range(num_epochs):
                # Validate
                print("Validating...")
                unet.eval()

                for name, valid_loader in zip(
                    ["val-A", "val-C", "val-B"],
                    [valid_loader_a, valid_loader_b, valid_loader_a2],
                ):
                    # for name, valid_ds in zip(["A"], [valid_ds_a]):
                    losses = []
                    images = []
                    for ims, masks, _ in valid_loader:
                        ims, masks = ims.to("cuda"), masks.to("cuda")
                        with torch.no_grad():
                            masks_out = unet(ims)[:, 0]
                            for m1, m2 in zip(masks_out, masks):
                                losses.append(metric(m1, m2).item())
                        for im, mask, mask_out in zip(ims[:, 0], masks, masks_out):
                            half_depth = im.size(1) // 2
                            im = im / im.max()
                            raw_im = im[:, half_depth].cpu().clone().expand(3, -1, -1)
                            pred_mask_im = raw_im.clone()
                            gt_mask_im = raw_im.clone()

                            # print(mask_out.dtype, mask.dtype)
                            pred_mask_im[:, mask_out[half_depth] > 0.5] = torch.tensor(
                                [0, 0, 1]
                            ).float()[:, None]
                            pred_mask_im[
                                :,
                                torch.logical_and(
                                    mask_out[half_depth] > 0.5, mask[half_depth] == 2
                                ),
                            ] = torch.tensor([0.8, 0, 0.8]).float()[:, None]
                            pred_mask_im[
                                :,
                                torch.logical_and(
                                    mask_out[half_depth] > 0.5, mask[half_depth] == 1
                                ),
                            ] = torch.tensor([0, 1, 1]).float()[:, None]
                            gt_mask_im[:, mask[half_depth] == 1] = torch.tensor(
                                [0, 1, 0]
                            ).float()[:, None]
                            gt_mask_im[:, mask[half_depth] == 2] = torch.tensor(
                                [1, 0, 0]
                            ).float()[:, None]

                            grid = make_grid([raw_im, pred_mask_im, gt_mask_im])
                            images.append(grid)

                    dice_mean = np.mean(losses)
                    metrics[name].append(dice_mean)
                    dice_std = np.std(losses)
                    writer.add_scalar(name, dice_mean, i)
                    writer.add_images(
                        f"validation {name} images", torch.stack(images), i
                    )
                print(
                    f"Validation epoch {j+1}, ",
                    f"validation set Loss mean : {dice_mean:.4f}, ",
                    f"Loss std : {dice_std:.4f}",
                )

                # Train
                print("Training...")
                unet.train()
                losses = []
                for im, mask, _ in tqdm(
                    train_loader,
                    total=n_samples_per_epoch // batch_size,
                    desc=f"Train epoch {i+1}",
                ):
                    optimizer.zero_grad()
                    im, mask = im.to("cuda"), mask.to("cuda")
                    mask_out = unet(im)[:, 0]
                    loss = loss(mask_out, mask)
                    loss.backward()
                    optimizer.step()

                    losses.append(copy.deepcopy(loss.detach().item()))
                    del im, mask, _, loss, mask_out
                dice_mean = np.mean(losses)
                writer.add_scalar("train loss", dice_mean, i)
                metrics["train"].append(dice_mean)

                # Evaluate
                print("Evaluate...")
                loss_ = 0
                for ims, masks, _ in valid_loader_early_stopping:
                    ims, masks = ims.to("cuda"), masks.to("cuda")
                    with torch.no_grad():
                        masks_out = unet(ims)[:, 0]
                        for m1, m2 in zip(masks_out, masks):
                            loss_ += metric(m1, m2).item()
                # early stopping, shouldn't be necessary
                if loss_ < best_loss:
                    best_loss = loss_
                    torch.save(
                        unet.state_dict(), os.path.join(writer.log_dir, "unet.pth")
                    )

            # Save training metrics
            pd.DataFrame(metrics).to_csv(
                os.path.join(writer.log_dir, "train_metrics.csv")
            )

            # Testing the model
            print("Testing...")
            unet = UNet3D(1, 1).to("cuda")
            unet.load_state_dict(torch.load(os.path.join(writer.log_dir, "unet.pth")))
            metadatas = []
            losses = []
            for ims, masks, infos in testing_loader:
                ims, masks = ims.to("cuda"), masks.to("cuda")
                with torch.no_grad():
                    masks_out = unet(ims)[:, 0]
                    for m1, m2, info in zip(masks_out, masks, unbatch_dict(infos)):
                        info["loss"] = metric(m1, m2).item()
                        metadatas.append(info)

            metadatas = default_collate(metadatas)
            infos_to_dataframe(metadatas).to_csv(
                os.path.join(writer.log_dir, "test_metrics.csv")
            )
