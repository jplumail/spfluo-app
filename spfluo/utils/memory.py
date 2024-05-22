from __future__ import annotations

import itertools
import logging
import math
from typing import Generator, Tuple

import numpy as np

logger = logging.getLogger("spfluo.utils.memory")


def _split_batch(
    max_batch: Tuple[int | None],
    shape: Tuple[int],
    offset: Tuple[int] = None,
) -> Generator[int | Tuple[int], None, None]:
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
                for o in _split_batch(
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


def split_batch(
    shape: Tuple[int],
    max_batch: int | None = None,
    use_memory: bool = False,
) -> Generator[int | Tuple[int], None, None]:
    if use_memory:
        raise NotImplementedError()

    if max_batch is None:
        yield from _split_batch(tuple([None for _ in range(len(shape))]), shape)
        return

    minibatch = find_minibatch(shape, max_batch)
    yield from _split_batch(
        tuple([0 for i in range(len(shape) - len(minibatch))]) + minibatch, shape
    )


if __name__ == "__main__":
    for o in split_batch((8, None, 10), (100, 120, 17)):
        print(o)
