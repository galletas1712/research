import numpy as np

from optimizer.circuit import sum_cons


def test_sum_cons():
    T = 2
    N = 3
    W = np.array(
        [
            [[0.2, 0, 0.4], [0.3, 0.5, 0.4], [0, 0.7, 0.2]],
            [
                [0.8, 0.9, 0.1],
                [0, 0, 0],
                [0.5, 0.1, 0.9],
            ],
        ]
    )
    assert np.isclose(
        sum_cons(T, N, W.reshape(T * N * N)), np.array([-0.5, 0.2, 0, 0.3, 0, 0])
    ).all()
