import numpy as np

from optimizer.circuit import amounts_out
from optimizer.core import net_out_transactions, no_back_forth_cons


def test_net_out_transactions():
    T = 1
    N = 3
    W_original = np.array(
        [
            [
                [0.3, 0.5, 0.2],
                [0.2, 0.5, 0.1],
                [0.5, 0.0, 0.7],
            ],
        ]
    )
    initial_alloc = np.array([1e4, 10, 0.2])
    initial_reserves = np.array(
        [
            [np.inf, 1e7 / 1000, 5e6 / 50000],
            [1e7, np.inf, 1e6 / 50000],
            [5e6, 1e6 / 1000, np.inf],
        ]
    )
    Y = 0.997

    W_netted = net_out_transactions(W_original, initial_alloc, initial_reserves, Y)
    amounts_received_original = amounts_out(
        W_original, initial_alloc, initial_reserves, Y
    )
    amounts_received_netted = amounts_out(W_netted, initial_alloc, initial_reserves, Y)
    assert (amounts_received_netted >= amounts_received_original).all()
    assert (no_back_forth_cons(T, N, W_netted.reshape(-1)) == 0).all()
