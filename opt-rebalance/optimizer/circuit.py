from typing import Tuple

import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)


def sum_cons(T: int, N: int, x: jnp.ndarray) -> jnp.ndarray:
    """Computes the column sums minus one for each weight matrix."""
    assert x.shape == (T * N * N,)
    W = x.reshape(T, N, N)
    return jnp.matmul(jnp.transpose(W, (0, 2, 1)).reshape(T * N, N), jnp.ones(N)) - 1


def next_iter(
    W_t: jnp.ndarray, I_t: jnp.ndarray, R_t: jnp.ndarray, Y: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes amounts of each token received and reserves after trades of one weight matrix."""
    N = len(I_t)

    assert I_t.shape == (N,)
    assert W_t.shape == (N, N)
    assert R_t.shape == (N, N)

    amounts_received_from = (
        jnp.nan_to_num(R_t, posinf=0)
        * W_t
        * I_t
        * Y
        / (jnp.transpose(R_t) + W_t * I_t * Y)
    )
    assert (amounts_received_from[jnp.isinf(R_t)] == 0).all()
    amounts_received_from = amounts_received_from + W_t * jnp.eye(N) * I_t
    amounts_received = jnp.matmul(amounts_received_from, jnp.ones(N))
    R_new = R_t - amounts_received_from + jnp.transpose(W_t * I_t)
    assert (R_new[jnp.isinf(R_t)] == jnp.inf).all() and (R_new > 0).all()

    return amounts_received, R_new


def amounts_out(
    W: jnp.ndarray, I_0: jnp.ndarray, R_0: jnp.ndarray, Y: float
) -> jnp.ndarray:
    """Computes amounts of tokens received after all trades given by a weight tensor."""
    assert len(W.shape) == 3 and W.shape[1] == W.shape[2]
    N = W.shape[1]
    assert I_0.shape == (N,)
    assert R_0.shape == (N, N)

    amounts_out = I_0
    R = R_0
    for t in range(len(W)):
        amounts_out, R = next_iter(W[t], amounts_out, R, Y)
    return amounts_out


def loss(
    W: jnp.ndarray,
    I_0: jnp.ndarray,
    O: jnp.ndarray,
    R_0: jnp.ndarray,
    P: jnp.ndarray,
    Y: float,
) -> float:
    """Computes the sum of squared differences between amounts of tokens after trades given by
    weight tensor and target allocation."""
    assert len(W.shape) == 3 and W.shape[1] == W.shape[2]
    N = W.shape[1]
    assert I_0.shape == (N,)
    assert O.shape == (N,)
    assert R_0.shape == (N, N)
    assert P.shape == (N,)

    diff = (amounts_out(W, I_0, R_0, Y) - O) * P / jnp.dot(O, P)
    return jnp.sum(jnp.square(diff))
