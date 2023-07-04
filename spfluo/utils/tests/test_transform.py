from spfluo.utils.transform import get_transform_matrix

import numpy as np


def test_get_transform_matrix_simple():
    H = get_transform_matrix((30,30,30), np.array([1.,2.,3.]), np.array([4.,5.,6.]))
    expected_result = np.array([
        [-0.48547846, -0.42291857,  0.7651474 , 20.57711966],
        [-0.8647801 ,  0.10384657, -0.4912955 , 37.65732099],
        [ 0.12832006, -0.90019763, -0.41614684, 37.72635389],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
    assert np.isclose(H, expected_result).all()

def test_get_transform_matrix_batch():
    output_shape = (30, 30, 30)
    N = 10
    rot = np.random.randn(N, 3)
    trans = np.random.randn(N, 3)

    matrices = get_transform_matrix(output_shape, rot, trans)
    for i in range(N):
        assert np.isclose(matrices[i], get_transform_matrix(output_shape, rot[i], trans[i])).all()