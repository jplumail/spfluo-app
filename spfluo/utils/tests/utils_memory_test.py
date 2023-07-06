import numpy as np

from spfluo.utils.memory import split_batch


def test_split_batch():
    shapes = [(10,), (100,), (100, 120, 7), (100, 120, 7), (100, 120, 7), (1000,)]
    batches = [(1,), (None,), (10, None, None), (21, 12, 6), (None, 12, 7), (2048,)]

    tables = [np.zeros(s, dtype=int) for s in shapes]

    for t, batch in zip(tables, batches):
        for idx in split_batch(batch, t.shape):
            if type(idx) is tuple:
                idx = [idx]
            slices = [slice(i, j) for i, j in idx]
            t[tuple(slices)] += 1
            assert all(
                [
                    (j - i) <= b if b is not None else True
                    for (i, j), b in zip(idx, batch)
                ]
            )
        assert (t == 1).all()
