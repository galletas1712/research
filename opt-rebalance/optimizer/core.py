from dataclasses import dataclass, field, astuple
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds, minimize

from optimizer.circuit import sum_cons, loss, amounts_out, next_iter


@dataclass
class TradeConditions:
    T: int
    N: int
    total_capital_denom: float
    fee_multiplier: float
    initial_alloc_proportion: np.ndarray
    target_alloc_proportion: np.ndarray
    initial_reserves: np.ndarray
    prices: np.ndarray = field(init=False)
    no_pair_indices: np.ndarray = field(init=False)
    initial_alloc: np.ndarray = field(init=False)
    target_alloc: np.ndarray = field(init=False)

    def __post_init__(self):
        assert (
            self.initial_alloc_proportion.shape == (self.N,)
            and self.initial_alloc_proportion.sum() == 1
        )
        assert (
            self.target_alloc_proportion.shape == (self.N,)
            and self.target_alloc_proportion.sum() == 1
        )
        assert (
            self.initial_reserves.shape == (self.N, self.N)
            and (np.diag(self.initial_reserves) == np.inf).all()
            and not np.isnan(self.initial_reserves[0]).any()
            and not np.isnan(self.initial_reserves[:, 0]).any()
        )
        assert 0 <= self.fee_multiplier <= 1

        self.prices = np.concatenate(
            ([1], (self.initial_reserves[0] / self.initial_reserves[:, 0])[1:])
        )
        self.no_pair_indices = np.isinf(self.initial_reserves) & ~np.eye(
            self.N, dtype=bool
        )

        initial_alloc_prediff = (
            self.initial_alloc_proportion * self.total_capital_denom / self.prices
        )
        target_alloc_prediff = (
            self.target_alloc_proportion * self.total_capital_denom / self.prices
        )
        self.initial_alloc = np.maximum(initial_alloc_prediff - target_alloc_prediff, 0)
        self.target_alloc = np.maximum(target_alloc_prediff - initial_alloc_prediff, 0)

        assert self.prices.shape == (self.N,)
        assert self.no_pair_indices.shape == (self.N, self.N)
        assert self.initial_alloc.shape == (self.N,)
        assert self.target_alloc.shape == (self.N,)
        assert np.isclose(
            np.dot(self.initial_alloc, self.prices),
            np.dot(self.target_alloc, self.prices),
        )

        # Check if reserves match prices
        for i in range(self.N):
            if i != 0:
                assert np.isclose(
                    self.initial_reserves[0][i] / self.initial_reserves[i][0],
                    self.prices[i],
                )


def no_pair_cons(
    T: int, N: int, x: np.ndarray, no_pair_indices: np.ndarray
) -> np.ndarray:
    """Generates a vector that should be zero if there are no weights on LP pairs."""
    assert x.shape == (T * N * N,)
    W = x.reshape(T, N, N)
    return W[:, no_pair_indices].flatten()


def no_back_forth_cons(T: int, N: int, x: np.ndarray) -> np.ndarray:
    """Generate a vector that should be zero if there are no trades back an forth within the same pair."""
    assert x.shape == (T * N * N,)
    W = x.reshape(T, N, N)

    # Select only non-redundant constraints
    triu_indices = np.triu_indices(N, 1)
    independent_cons_indices = (
        np.repeat(np.arange(0, T), int(N * (N - 1) / 2)),
        np.tile(triu_indices[0], T),
        np.tile(triu_indices[1], T),
    )

    check = (np.transpose(W, (0, 2, 1)) * W)[independent_cons_indices]

    return check


def net_out_transactions(
    W: np.ndarray,
    initial_alloc: np.ndarray,
    initial_reserves: np.ndarray,
    fee_multiplier: float,
) -> np.ndarray:
    """Performs postprocessing to ensure no back and forth trades are made on the same pair."""
    assert len(W.shape) == 3 and W.shape[1] == W.shape[2]
    N = W.shape[1]
    assert initial_alloc.shape == (N,)
    assert initial_reserves.shape == (N, N)

    I_t = initial_alloc
    R_t = initial_reserves
    W = W.copy()

    for t in range(W.shape[0]):
        P_t = np.concatenate([[1], R_t[0, 1:] / R_t[1:, 0]])
        for i in range(N):
            for j in range(i + 1, N):
                if W[t, i, j] * P_t[j] * I_t[j] > W[t, j, i] * P_t[i] * I_t[i]:
                    W[t, i, j] -= W[t, j, i] * P_t[i] * I_t[i] / (P_t[j] * I_t[j])
                    W[t, i, i] += W[t, j, i]
                    W[t, j, j] += W[t, j, i] * P_t[i] * I_t[i] / (P_t[j] * I_t[j])
                    W[t, j, i] = 0  # needs to go last bc mutation
                elif W[t, j, i] * P_t[i] * I_t[i] > W[t, i, j] * P_t[j] * I_t[j]:
                    W[t, j, i] -= W[t, i, j] * P_t[j] * I_t[j] / (P_t[i] * I_t[i])
                    W[t, j, j] += W[t, i, j]
                    W[t, i, i] += W[t, i, j] * P_t[j] * I_t[j] / (P_t[i] * I_t[i])
                    W[t, i, j] = 0
        I_t, R_t = next_iter(W[t], I_t, R_t, fee_multiplier)

    return W


def postprocess(
    W: np.ndarray, eps: float, trade_conditions: TradeConditions
) -> np.ndarray:
    """Performs postprocessing on weights from optimizer."""
    assert W.shape == (trade_conditions.T, trade_conditions.N, trade_conditions.N)

    W = W.copy()

    # Ignore swaps that are too small to make sense
    W[W < eps] = 0

    # Net out transactions in the same pair
    W = net_out_transactions(
        W,
        trade_conditions.initial_alloc,
        trade_conditions.initial_reserves,
        trade_conditions.fee_multiplier,
    )

    # Ensure sum constraint is actually satisfied
    W = W / np.sum(W, axis=1, keepdims=True)

    return W


def optimize_trades_slsqp(
    trade_conditions: TradeConditions,
    x0: np.ndarray = None,
    eps: float = 1e-3,
    optimizer_ftol: float = 1e-9,
    optimizer_max_iters: int = 100,
    verbose: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Runs SLSQP to optimize trades, given initial trade conditions, and optionally random starting weights,
    precision tolerance, and SLSQP parameters."""
    (
        T,
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

    no_pair_indices = np.isinf(initial_reserves) & ~jnp.eye(N, dtype=bool)

    # If no initial start is provided, use flattened identity tensor
    if x0 is None:
        x0 = np.tile(np.eye(N), (T, 1)).flatten()

    objective = lambda x: loss(
        jnp.array(x.reshape(T, N, N)),
        jnp.array(initial_alloc),
        jnp.array(target_alloc),
        jnp.array(initial_reserves),
        jnp.array(prices),
        fee_multiplier,
    )

    # Force weights at nonexistent pairs to be zero
    bounds = Bounds(
        jnp.zeros(T * N * N), jnp.tile(1 - no_pair_indices.flatten().astype(int), T)
    )

    minimize_result = minimize(
        objective,
        jnp.array(x0),
        method="SLSQP",
        jac=jax.grad(objective),
        constraints=[
            {
                "type": "eq",
                "fun": lambda x: sum_cons(T, N, x),
                "jac": lambda xp: jax.jacrev(lambda x: sum_cons(T, N, x))(xp),
            }
        ],
        bounds=bounds,
        options={
            "ftol": optimizer_ftol,
            "maxiter": optimizer_max_iters,
            "disp": verbose,
        },
    )

    if minimize_result.status != 0 and minimize_result.status != 9:
        print(minimize_result.status)
        raise Exception("Failed to find optimal trade")

    W = np.array(minimize_result.x).reshape(T, N, N)
    W = postprocess(W, eps, trade_conditions)
    amounts_received = amounts_out(W, initial_alloc, initial_reserves, fee_multiplier)
    nav_loss = np.dot(target_alloc - amounts_received, prices)

    return nav_loss, amounts_received, W
