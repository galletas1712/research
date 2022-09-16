import time
from dataclasses import astuple
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from optimizer.core import amounts_out, optimize_trades_slsqp, TradeConditions


def random_W0(
    T: int, N: int, no_pair_indices: np.ndarray, random_seed: int
) -> np.ndarray:
    """Creates a random initialization within feasible region."""
    rng = np.random.default_rng(random_seed)

    # Create mask to ensure no back and forth trades and no trades to nonexistent pairs
    mask = np.zeros((T, N, N))
    for t in range(T):
        mask[t][np.triu_indices(N)] = rng.binomial(1, 0.5, int(N * (N + 1) / 2))
        mask[t][np.tril_indices(N)] = 1 - np.transpose(mask[t])[np.tril_indices(N)]
        mask[t][
            np.diag_indices(N)
        ] = 1  # NOTE: if this is random, some columns may have sum 0 which creates nans
    mask[:, no_pair_indices] = 0

    # Create masked and normalized random init
    W0 = rng.uniform(0, 1, (T, N, N)) * mask
    W0 = W0 / np.sum(W0, axis=1, keepdims=True)
    assert not np.isnan(W0).any()

    return W0


def naive_solution(
    trade_conditions: TradeConditions,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Swap everything to denominated asset and swap to target allocation. Assumes all assets have pair with denominated
    asset."""
    (
        _,
        N,
        _,
        fee_multiplier,
        _,
        _,
        initial_reserves,
        prices,
        _,
        initial_alloc,
        target_alloc,
    ) = astuple(trade_conditions)
    # All assets must have pair with denominated asset
    assert (
        ~np.isinf(initial_reserves[1:, 0]) & ~np.isinf(initial_reserves[0, 1:])
    ).all()

    W = np.stack(
        [
            np.vstack([np.ones((1, N)), np.zeros((N - 1, N))]),
            np.hstack(
                [
                    (target_alloc * prices / np.sum(target_alloc * prices)).reshape(
                        -1, 1
                    ),
                    np.zeros((N, N - 1)),
                ]
            ),
        ],
        axis=0,
    )

    # Check that there are no trades back and forth of the same pair
    check = np.transpose(W, (0, 2, 1)) * W
    assert (((check > 0) | np.eye(N, dtype=bool)) == np.eye(N, dtype=bool)).all()

    amounts_received = amounts_out(W, initial_alloc, initial_reserves, fee_multiplier)
    nav_loss = np.dot(target_alloc - amounts_received, prices)
    return nav_loss, amounts_received, W


def optimize_trades_multistart(
    num_starts: int,  # Ideal number is around 10 for current large test conditions
    trade_conditions: TradeConditions,
    eps: float = 1e-3,
    optimizer_ftol: float = 1e-9,
    optimizer_max_iters: int = 100,
    verbose: bool = False,
    random_seed: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Runs optimizer_trades_slsqp from num_starts random init weights and takes the best of all results, including
    the naive solution. This function is only for testing purposes and will be deprecated for a parallel version
    in production."""
    (
        T,
        N,
        _,
        _,
        _,
        _,
        initial_reserves,
        _,
        _,
        initial_alloc,
        target_alloc,
    ) = astuple(trade_conditions)

    no_pair_indices = jnp.isinf(initial_reserves) & ~jnp.eye(N, dtype=bool)

    rng = np.random.default_rng(random_seed)
    best_nav_loss, best_amounts_received, best_W = naive_solution(trade_conditions)

    total_time_elapsed = 0
    for _ in range(num_starts):
        start_time = time.process_time()

        W0 = random_W0(T, N, no_pair_indices, rng.integers(int(1e9)))
        initial_alloc = jnp.maximum(initial_alloc - target_alloc, 0)
        target_alloc = jnp.maximum(target_alloc - initial_alloc, 0)
        nav_loss, amounts_received, W, = optimize_trades_slsqp(
            trade_conditions,
            W0.flatten(),
            eps,
            optimizer_ftol,
            optimizer_max_iters,
            verbose,
        )
        if nav_loss < best_nav_loss:
            best_nav_loss = nav_loss
            best_amounts_received = amounts_received
            best_W = W

        end_time = time.process_time()
        elapsed_time = end_time - start_time
        total_time_elapsed += elapsed_time

        if verbose:
            print("Elapsed time:", elapsed_time)
            print("NAV loss:", nav_loss)

    if verbose:
        print("Total time elapsed:", total_time_elapsed)

    return best_nav_loss, best_amounts_received, best_W
