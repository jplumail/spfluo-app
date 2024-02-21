from __future__ import annotations

import inspect
import itertools
import logging
import math
from typing import Generator, Tuple

import numpy as np
import torch
from torch import cuda

from spfluo.utils.debug import arg_pretty_repr

try:
    from pynvml.smi import nvidia_smi

    pynvml_installed = True
except ModuleNotFoundError:
    pynvml_installed = False

try:
    import psutil

    psutil_installed = True

except ModuleNotFoundError:
    psutil_installed = False

NVSMI = None


logger = logging.getLogger("spfluo.utils.memory")


def format_memory_size(size_in_bytes):
    # Define the suffixes for each magnitude
    for unit in ["Bytes", "KB", "MB", "GB"]:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} TB"


def nvidia_free_memory() -> int:
    """
    Calls nvidia's nvml library and queries available GPU memory.
    Currently the function only works with 1 GPU.

    Returns
    -------

    Free GPU memory in terms of bytes.
    """

    global NVSMI
    if NVSMI is None:
        NVSMI = nvidia_smi.getInstance()

    assert NVSMI is not None
    query = NVSMI.DeviceQuery("memory.free")

    # Only works on one GPU as of now.
    gpu = query["gpu"][0]["fb_memory_usage"]

    if gpu["unit"] == "MiB":
        unit = 2**20
    else:
        unit = None

    free = gpu["free"]

    assert unit is not None
    return free * unit


def torch_free_memory() -> int:
    """
    Calls torch's memory statistics to calculate the amount of GPU memory unused.
    Currently the function only works with 1 GPU.

    Returns
    -------

    Reserved GPU memory in terms of bytes.
    """

    if not cuda.is_available():
        return 0

    # Only works on one GPU as of now.

    reserved_memory = cuda.memory_reserved(0)
    active_memory = cuda.memory_allocated(0)
    unused_memory = reserved_memory - active_memory
    return unused_memory


def free_memory_gpu() -> int | None:
    """
    The amount of free GPU memory that can be used.

    Returns
    -------

    Unused GPU memory, or None if no GPUs are available.
    """

    if pynvml_installed and cuda.is_available():
        return nvidia_free_memory() + torch_free_memory()
    else:
        return None


def free_memory_cpu():
    if psutil_installed:
        return psutil.virtual_memory().available
    else:
        return 2 * 2**10  # assume 2GB available


def maximum_batch(total_memory, func: str, *func_args):
    if func == "convolution_matching_poses_grid":
        reference, volumes, _, potential_poses = func_args
        D, H, W = reference.shape[:3]
        N, _, _, _ = volumes.size()
        M, _ = potential_poses.size()
        shape = (N, M)
        dtype_bytes = torch.finfo(reference.dtype).bits / 8
        total_batch = (
            total_memory
            * 4
            * (32**3)
            * 128
            * 100
            / (12000 * (2**20) * dtype_bytes * H * W * D)
        )
        max_batch_ = math.floor(total_batch**0.5)
        if max_batch_ > N and max_batch_ > M:
            max_batch = (None, None)
        elif max_batch_ > N:
            if (mbatch := math.floor(total_batch / N)) > M:
                max_batch = (None, None)
            else:
                max_batch = (None, mbatch)
        elif max_batch_ > M:
            if (nbatch := math.floor(total_batch / M)) > N:
                max_batch = (None, None)
            else:
                max_batch = (nbatch, None)
        else:
            max_batch = (max_batch_, max_batch_)

    if func == "convolution_matching_poses_refined":
        reference, volumes, _, potential_poses = func_args
        D, H, W = reference.shape[:3]
        N, _, _, _ = volumes.size()
        _, M, _ = potential_poses.size()
        shape = (N, M)
        dtype_bytes = torch.finfo(reference.dtype).bits / 8
        total_batch = (
            total_memory
            * 4
            * (32**3)
            * 128
            * 100
            / (12000 * (2**20) * dtype_bytes * H * W * D)
        )
        max_batch_ = math.floor(total_batch**0.5)
        if max_batch_ > N and max_batch_ > M:
            max_batch = (None, None)
        elif max_batch_ > N:
            if (mbatch := math.floor(total_batch / N)) > M:
                max_batch = (None, None)
            else:
                max_batch = (None, mbatch)
        elif max_batch_ > M:
            if (nbatch := math.floor(total_batch / M)) > N:
                max_batch = (None, None)
            else:
                max_batch = (nbatch, None)
        else:
            max_batch = (max_batch_, max_batch_)

    if func == "reconstruction_L2":
        volumes, psf, poses, lambda_ = func_args
        D, H, W = volumes.size()[-3:]
        M, k, N, _ = poses.size()
        dtype_bytes = torch.finfo(poses.dtype).bits / 8
        total_batch = (
            total_memory
            * 4
            * (128**3)
            * 5
            * 8
            / (5000 * (2**20) * dtype_bytes * N * H * W * D)
        )
        max_batch = (math.floor(total_batch**0.5), math.floor(total_batch**0.5))
        shape = (M, k)

    if func == "gd_importance_sampling_3d":
        image_shape, N = func_args
        shape = (N,)
        max_batch = (None,)
    return max_batch, shape


def split_batch_func(func, *func_args, total_memory=None, max_batch=None):
    func = inspect.stack()[1].function
    logger.info(
        f"split_batch_func for func {func} with args ("
        f"{' '.join([arg_pretty_repr(arg) for arg in func_args])})."
    )
    if total_memory is None:
        total_memory = (
            free_memory_gpu()
            if hasattr(func_args[0], "is_cuda") and func_args[0].is_cuda
            else free_memory_cpu()
        )
        logger.debug(f"Total free memory: {format_memory_size(total_memory)}")
    max_batch_inferred, shape = maximum_batch(total_memory, func, *func_args)
    if max_batch is None:
        max_batch = max_batch_inferred
    logger.debug(f"Maximum batch size: {max_batch}")
    logger.debug(f"Total shape: {shape}")
    yield from split_batch(max_batch, shape)


def split_batch(
    max_batch: Tuple[int | None],
    shape: Tuple[int],
    offset: Tuple[int] = None,
) -> Generator[int | Tuple[int], None, None]:
    # max_batch = maximum_batch(memory, total_memory)

    if type(max_batch) is tuple:
        assert len(max_batch) == len(shape)
        max_batch = np.array(
            [mb if mb is not None else shape[i] for i, mb in enumerate(max_batch)]
        )
    else:
        max_batch = np.array([max_batch], dtype=int)

    max_batch = np.maximum(max_batch, 1)
    batch_size = np.maximum(2 ** (np.floor(np.log2(max_batch))), 1)
    batch_size = batch_size.astype(int)
    (times, remain_shape) = np.divmod(shape, batch_size)

    if offset is None:
        offset = np.zeros((batch_size.shape[0]), dtype=int)
    ranges = [range(t) for t in times]
    for x in itertools.product(*ranges):
        start = offset + batch_size * np.array(x)
        end = start + batch_size
        out = list(zip(start, end))
        if len(out) == 1:
            logger.debug(
                f"Minibatch summary: minibatch={out[0]}, {max_batch=}, {shape=}"
            )
            yield out[0]
        else:
            logger.debug(f"Minibatch summary: minibatch={out}, {max_batch=}, {shape=}")
            yield out

    if remain_shape.sum() > 0:
        rectangle = times * batch_size
        dims = remain_shape.nonzero()[0]
        if isinstance(shape, int):
            shape = np.array([shape])
        for i in range(1, len(dims) + 1):
            for combination in itertools.combinations(dims, i):
                combination = np.asarray(combination)
                new_shape = rectangle.copy()
                new_shape[combination] = (
                    np.asarray(shape)[combination] - rectangle[combination]
                )
                new_offset = offset.copy()
                new_offset[combination] += rectangle[combination]
                for o in split_batch(
                    tuple(np.minimum(max_batch, new_shape)),
                    tuple(new_shape),
                    new_offset,
                ):
                    logger.debug(
                        f"Minibatch summary: minibatch={o}, {max_batch=}, {shape=}"
                    )
                    yield o


def find_minibatch(shape: tuple[int], max_size: int) -> tuple[int]:
    if len(shape) == 0:
        return ()
    cumproduct = np.cumprod(np.flip(np.asarray(shape)))
    i = np.sum(cumproduct <= max_size)
    if i == 0:
        return (max_size,)
    else:
        return (
            find_minibatch(shape[:-i], math.floor(max_size / cumproduct[i - 1]))
            + shape[-i:]
        )


def split_batch2(
    shape: Tuple[int],
    max_batch: int | None = None,
    use_memory: bool = False,
) -> Generator[int | Tuple[int], None, None]:
    if use_memory:
        raise NotImplementedError()

    if max_batch is None:
        yield from split_batch(tuple([None for _ in range(len(shape))]), shape)
        return

    minibatch = find_minibatch(shape, max_batch)
    yield from split_batch(
        tuple([0 for i in range(len(shape) - len(minibatch))]) + minibatch, shape
    )


if __name__ == "__main__":
    for o in split_batch((8, None, 10), (100, 120, 17)):
        print(o)
