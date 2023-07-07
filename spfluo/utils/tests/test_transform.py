import numpy as np

from spfluo.utils.transform import distance_poses, get_transform_matrix


def test_get_transform_matrix_simple():
    H = get_transform_matrix(
        (30, 30, 30),
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        convention="ZXZ",
        degrees=False,
    )
    expected_result = np.array(
        [
            [-0.48547846, -0.42291857, 0.7651474, 20.57711966],
            [-0.8647801, 0.10384657, -0.4912955, 37.65732099],
            [0.12832006, -0.90019763, -0.41614684, 37.72635389],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.isclose(H, expected_result).all()


def test_get_transform_matrix_batch():
    output_shape = (30, 30, 30)
    N = 10
    rot = np.random.randn(N, 3)
    trans = np.random.randn(N, 3)

    matrices = get_transform_matrix(output_shape, rot, trans)
    for i in range(N):
        assert np.isclose(
            matrices[i], get_transform_matrix(output_shape, rot[i], trans[i])
        ).all()


def test_distance_poses():
    p1 = np.asarray([90, 90, 0, 1, 0, 0], dtype=float)
    p2 = np.asarray([-90, 90, 0, 0, 1, 0], dtype=float)
    angle, t = distance_poses(p1, p2)

    assert np.isclose(angle, 180.0)
    assert np.isclose(t, 2.0**0.5)
