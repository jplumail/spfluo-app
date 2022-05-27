from __future__ import annotations
import itertools

import math
import numpy as np
from typing import Generator, Tuple
from pynvml.smi import nvidia_smi
from torch import cuda


NVSMI = None


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

    if gpu["unit"] == 'MiB':
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


def free_memory() -> int | None:
    """
    The amount of free GPU memory that can be used.

    Returns
    -------

    Unused GPU memory, or None if no GPUs are available.
    """

    if cuda.is_available():
        return nvidia_free_memory() + torch_free_memory()
    else:
        return None


def maximum_batch(memory, total_memory: int | None = None) -> int | None:
    # batch * x + no_batch = unused_memoroy
    if total_memory is None:
        total_memory = free_memory()

    if total_memory is None:
        return None

    return (total_memory - memory.no_batch) // memory.batch


def split_batch(
    max_batch: Tuple[int|None], shape: Tuple[int], total_memory: int | None = None, offset: Tuple[int] = None
) -> Generator[int | Tuple[int], None, None]:
    #max_batch = maximum_batch(memory, total_memory)


    if type(max_batch) is tuple:
        assert len(max_batch) == len(shape)
        max_batch = np.array([mb if mb is not None else shape[i] for i, mb in enumerate(max_batch)])
    else:
        max_batch = np.array([max_batch], dtype=int)

    batch_size = 2 ** (np.floor(np.log2(max_batch)))
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
            yield out[0]
        else:
            yield out

    if remain_shape.sum() > 0:
        rectangle = times * batch_size
        dims = remain_shape.nonzero()[0]
        if type(shape) is int:
            shape = np.array([shape])
        for i in range(1, len(dims)+1):
            for combination in itertools.combinations(dims, i):
                combination = np.asarray(combination)
                new_shape = rectangle.copy()
                new_shape[combination] = np.asarray(shape)[combination] - rectangle[combination]
                new_offset = offset.copy()
                new_offset[combination] += rectangle[combination]
                for o in split_batch(tuple(np.minimum(max_batch, new_shape)), tuple(new_shape), None, new_offset):
                    yield o


if __name__ == '__main__':
    for o in split_batch((8,None,10), (100,120,17)):
        print(o)