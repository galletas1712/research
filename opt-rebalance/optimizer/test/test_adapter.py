import numpy as np

from optimizer.adapter import Trade, gen_trades


def test_gen_trades_two_assets_one_trade_max():
    W = np.array(
        [
            [1, 0.7],
            [0, 0.3],
        ]
    )
    initial_alloc = np.array([10000, 20])
    trades = gen_trades(W, initial_alloc)
    assert trades == [Trade(1, 0, 14)]


def test_gen_trades_bigger():
    # DAI, ETH, BTC
    W = np.array(
        [
            [1.0, 0.6, 0.5],
            [0.0, 0.4, 0.4],
            [0.0, 0.0, 0.1],
        ],
    )
    initial_alloc = np.array([10000, 20, 1])
    trades = gen_trades(W, initial_alloc)
    assert trades == [
        Trade(1, 0, 12),
        Trade(2, 0, 0.5),
        Trade(2, 1, 0.4),
    ]
