from dataclasses import astuple

import numpy as np

from optimizer.core import no_pair_cons, no_back_forth_cons
from optimizer.multistart import (
    naive_solution,
    optimize_trades_multistart,
)


def run_multistart(
    num_runs,
    trade_conditions,
):
    (
        T,
        N,
        _,
        fee_multiplier,
        _,
        _,
        initial_reserves,
        prices,
        no_pair_indices,
        initial_alloc,
        target_alloc,
    ) = astuple(trade_conditions)
    naive_nav_loss, naive_amounts_received, naive_W = naive_solution(trade_conditions)
    print("Naive NAV loss:", naive_nav_loss)

    nav_loss, amounts_received, W = optimize_trades_multistart(
        num_runs,
        trade_conditions,
        verbose=True,
    )

    print("Sum constraint:", np.sum(W, axis=1, keepdims=True))
    if no_pair_indices is not None:
        print("No pair constraint:", no_pair_cons(T, N, W.flatten(), no_pair_indices))
    print("No back and forth constraint:", no_back_forth_cons(T, N, W.flatten()))

    print("Amounts received:", amounts_received)
    print("Target alloc:", target_alloc)
    print("Diffs:", (amounts_received - target_alloc) * prices)
    total_abs_diff = np.dot(np.abs(amounts_received - target_alloc), prices)
    trade_size = np.dot(initial_alloc, prices)
    print("Trade Size:", trade_size)
    print("Total Abs Diff / Trade Size:", total_abs_diff / trade_size)
    print("NAV loss:", np.dot(target_alloc - amounts_received, prices))
    print(
        "NAV loss / Trade Size:",
        np.dot(target_alloc - amounts_received, prices) / trade_size,
    )


# Small weight matrix space to optimize over - synthetic market conditions considering only 4 assets
def test_small(small_trade_conditions):
    run_multistart(10, small_trade_conditions)


# Larger weight matrix space to optimize over - synthetic market conditions considering 10 assets
def test_big(big_trade_conditions):
    run_multistart(10, big_trade_conditions)


# Small weight matrix to optimize over - real market conditions considering 4 assets
def test_real_conditions(real_trade_conditions):
    run_multistart(10, real_trade_conditions)
