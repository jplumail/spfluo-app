import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..training import EfficientNet, make_U_mask
from ..training.prepare_fs_data import get_patch_paths
from ..utils import load_array, summary

# +------------------------------------------------------------------------------------------+ #
# |                                          HELPERS                                         | #
# +------------------------------------------------------------------------------------------+ #


def load_network(
    network: str, dim: int, num_classes: int, checkpoint_path: str, device: torch.device
) -> torch.nn.Module:
    network = EfficientNet.from_name(network, dim=dim, num_classes=num_classes).to(
        device
    )
    checkpoint = torch.load(checkpoint_path)
    network.load_state_dict(checkpoint["network_state_dict"])
    network = network.to(device)
    network = network.eval()
    return network


def make_patches(
    image: torch.Tensor, patch_size: Tuple[int], stride: Tuple[int], batch_size: int = 1
):
    pz, py, px = patch_size
    sz, sy, sx = stride

    image_batch_buffer = torch.empty((batch_size,) + patch_size, dtype=image.dtype)
    coords_batch_buffer = torch.empty((batch_size,) + (3,), dtype=torch.int64)
    i = 0
    for z in range(0, image.shape[0] - pz + 1, sz):
        for y in range(0, image.shape[1] - py + 1, sy):
            for x in range(0, image.shape[2] - px + 1, sx):
                image_batch_buffer[i] = image[z : z + pz, y : y + py, x : x + px]
                coords_batch_buffer[i] = torch.as_tensor(
                    (z + pz // 2, y + py // 2, x + px // 2), dtype=torch.int64
                )
                i += 1
                if i == batch_size:
                    yield (
                        torch.clone(image_batch_buffer),
                        torch.clone(coords_batch_buffer),
                    )
                    i = 0
    if i > 0:
        yield torch.clone(image_batch_buffer[:i]), torch.clone(coords_batch_buffer)


def flatten_patch_index_to_image_coord(
    index: int, unfold_shape: torch.Size, stride: Tuple[int]
) -> Tuple[int]:
    Z, Y, X, pz, py, px = unfold_shape
    sz, sy, sx = stride
    z = index // (Y * X)
    yx = index % (Y * X)
    y = yx // X
    x = yx % X
    return z * sz + pz // 2, y * sy + py // 2, x * sx + px // 2


# +------------------------------------------------------------------------------------------+ #
# |                                  MAIN PREDICTION FUNCTION                                | #
# +------------------------------------------------------------------------------------------+ #


@torch.no_grad()
def predict_one_image_picking(
    image: torch.Tensor,
    patch_size: Tuple[int],
    dim: int,
    network: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    stride: Tuple[int] = None,
    predict_on_u_mask: bool = False,
    progress_bar: bool = True,
) -> Tuple[np.ndarray]:
    patches_generator = make_patches(image, patch_size, stride, batch_size=batch_size)
    if predict_on_u_mask:
        # make_u_mask works on numpy array so some casting overhead is required.
        u_mask = make_U_mask(image.numpy())
        u_mask = torch.as_tensor(u_mask, dtype=torch.float32)
        u_mask_patches_generator = make_patches(
            u_mask, patch_size, stride, batch_size=batch_size
        )
    else:
        u_mask_patches_generator = (None for _ in iter(int, 1))  # infinite None
    coords: list[tuple[int, int, int]] = []
    scores: list[float] = []
    for (patches, patches_coords), (u_mask_patches, _) in zip(
        patches_generator, u_mask_patches_generator
    ):
        # Filter out patches not on U masks
        if u_mask_patches is not None:
            pz, py, px = patch_size
            filtered_indices = torch.where(
                u_mask_patches[:, pz // 2, py // 2, px // 2] > 0
            )[0]
        else:
            filtered_indices = torch.arange((patches.size(0),), dtype=torch.int64)
        patches = patches[filtered_indices]
        patches_coords = patches_coords[filtered_indices]

        if patches.size(0) > 0:
            patches = torch.stack(
                [(p - p.min()) / (p.max() - p.min()) for p in patches]
            )
            patches = patches.unsqueeze(1)  # add channels dim

            # Project
            if dim == 2:
                patches = patches.sum(axis=-3)

            # Predict
            inputs = patches.to(device)
            outputs = network(inputs)
            positive_indices = torch.where(outputs > 0)[0]
            for batch_index in positive_indices:
                scores.append(outputs[batch_index].item())
                coords.append(patches_coords[batch_index].tolist())
    return np.array(coords), np.array(scores)


@torch.no_grad()
def predict_patches_tilt(
    image: str,
    patches_path: List[str],
    rootdir: str,
    extension: str,
    dim: int,
    network: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    progress_bar: bool = True,
) -> Tuple[np.ndarray]:
    test_paths_image = list(
        filter(
            lambda x: "_".join(os.path.basename(x).split("_")[:-1]) + "." + extension
            == image,
            patches_path,
        )
    )
    test_paths_image.sort(
        key=lambda x: int(
            os.path.basename(x).split("_")[-1].split(".")[0],
        )
    )
    patch_names = list(map(lambda x: os.path.basename(x), test_paths_image))
    patches = []
    for crop_path in test_paths_image:
        patch_path = os.path.join(rootdir, crop_path)
        arr = load_array(patch_path)
        if arr.dtype == np.uint16:
            arr = arr.astype(np.uint8)
        patch = torch.as_tensor(arr, dtype=torch.float32)
        patches.append(patch)
    patches = torch.stack(patches, dim=0)

    if dim == 2:
        patches = patches.sum(axis=1)
    flatten_patches = patches.view(patches.size(0), -1)
    (min_patch, _), (max_patch, _) = (
        flatten_patches.min(dim=1),
        flatten_patches.max(dim=1),
    )
    min_patch = min_patch.view(-1, 1, 1)
    max_patch = max_patch.view(-1, 1, 1)
    patches = (patches - min_patch) / (max_patch - min_patch)
    patches = patches.unsqueeze(1)

    num_batches = patches.size(0) // batch_size
    generator = tqdm(range(num_batches)) if progress_bar else range(num_batches)
    scores, labels = [], []
    for i in generator:
        inputs = patches[i * batch_size : (i + 1) * batch_size].to(device)
        outputs = network(inputs)
        for batch_index in range(len(outputs)):
            score = outputs[batch_index].cpu().numpy()
            scores.append(score)
            label = score.argmax()
            labels.append(label)

    if patches.size(0) % batch_size != 0:
        inputs = patches[num_batches * batch_size :].to(device)
        outputs = network(inputs)
        for batch_index in range(len(outputs)):
            score = outputs[batch_index].cpu().numpy()
            scores.append(score)
            label = score.argmax()
            labels.append(label)

    scores = np.stack(scores, axis=0)
    return np.array(labels), scores, np.array(patch_names)


# +------------------------------------------------------------------------------------------+ #
# |                                       EXTERNAL CALL                                      | #
# +------------------------------------------------------------------------------------------+ #


def predict_picking(
    network: str,
    checkpoint_path: str,
    rootdir: str,
    predict_on_u_mask: bool,
    patch_size: Tuple[int],
    dim: int,
    stride: Tuple[int],
    batch_size: int,
    extension: str,
    testset_size: int = None,
    image_name: str = None,
    output_dir: str = None,
    downscale: float = 1.0,
) -> Dict[str, Tuple[np.ndarray]]:
    """If image_name is None, predict whole folder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = tuple(np.rint(np.array(patch_size) / downscale).astype(int))
    stride = (
        np.array(patch_size) // 4
        if stride is None
        else tuple(np.rint(np.array(stride) / downscale).astype(int))
    )
    summary_output = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        summary_output = os.path.join(output_dir, "prediction_config.txt")
    summary(locals(), title="PREDICTION", output=summary_output)
    network = load_network(network, dim, 1, checkpoint_path, device)
    predict_args = [
        patch_size,
        dim,
        network,
        device,
        batch_size,
        stride,
        predict_on_u_mask,
    ]
    if image_name is not None:
        images_names = [image_name]
    else:
        images_names = sorted(
            list(filter(lambda x: x.endswith(extension), os.listdir(rootdir)))
        )
        images_names = images_names[:testset_size]
    predictions = {}
    for image_name in images_names:
        image_path = os.path.join(rootdir, image_name)
        image = torch.as_tensor(load_array(image_path), dtype=torch.float32)
        predictions[image_name] = predict_one_image_picking(image, *predict_args)
    if output_dir is not None:
        with open(os.path.join(output_dir, "predictions.pickle"), "wb") as file:
            pickle.dump(predictions, file)
    return predictions


def predict_tilt(
    rootdir: str,
    network: str,
    checkpoint_path: str,
    crop_dir: str,
    dim: int,
    batch_size: int,
    extension: str,
    testset_size: int = None,
    image_name: str = None,
    output_dir: str = None,
) -> Dict[str, Tuple[np.ndarray]]:
    """If image_name is None, predict whole folder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_output = None
    if output_dir is not None:
        summary_output = os.path.join(output_dir, "prediction_config.txt")
        os.makedirs(output_dir, exist_ok=True)
    summary(locals(), title="PREDICTION", output=summary_output)
    network = load_network(network, dim, 3, checkpoint_path, device)
    predict_args = [extension, dim, network, device, batch_size]

    if image_name is not None:
        images_names = [image_name]
    else:
        images_names = sorted(
            list(filter(lambda x: x.endswith(extension), os.listdir(rootdir)))
        )
        images_names = images_names[:testset_size]

    test_paths = get_patch_paths(crop_dir)
    predictions = {}
    for image_name in images_names:
        predictions[image_name] = predict_patches_tilt(
            image_name, test_paths, crop_dir, *predict_args
        )

    if output_dir is not None:
        with open(os.path.join(output_dir, "predictions.pickle"), "wb") as file:
            pickle.dump(predictions, file)

    return predictions
