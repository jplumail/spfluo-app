import numpy as np

from spfluo.utils.memory import split_batch


def test_split_batch():
    shapes = [(10,), (100,), (100, 120, 7), (100, 120, 7), (100, 120, 7), (1000,)]
    max_batch_sizes = [1, None, 10, 21 * 12 * 6, 12 * 7, 2048]

    tables = [np.zeros(s, dtype=int) for s in shapes]

    for t, max_batch in zip(tables, max_batch_sizes):
        for idx in split_batch(t.shape, max_batch):
            if type(idx) is tuple:
                idx = [idx]
            slices = [slice(i, j) for i, j in idx]
            t[tuple(slices)] += 1
            assert all(
                [
                    (j - i) <= max_batch if max_batch is not None else True
                    for (i, j) in idx
                ]
            )
        assert (t == 1).all()
