import numpy as np
import pytest

from optimizer.adapter import init_network, init_token_contracts, get_decimals, init_pair_contracts, get_reserves
from optimizer.core import TradeConditions
from optimizer.multistart import random_W0


@pytest.fixture
def factory_address():
    return "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"


@pytest.fixture
def symbols():
    return ["ETH", "USDC", "BOND", "FRAX"]


@pytest.fixture
def token_addresses():
    return [
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # ETH
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        "0x0391d2021f89dc339f60fff84546ea23e337750f",  # BOND
        "0x853d955acef822db058eb8505911ed77f175b99e",  # FRAX
    ]


@pytest.fixture
def real_trade_conditions(factory_address, symbols, token_addresses):
    T = 2
    N = 4
    total_capital_denom = 1e6
    fee_multiplier = 0.997
    initial_alloc_proportion = np.array([0.3, 0.2, 0.2, 0.3])
    target_alloc_proportion = np.array([0.2, 0.3, 0, 0.5])
    init_network()
    token_contracts = init_token_contracts(symbols, token_addresses)
    decimals = get_decimals(token_contracts)
    pair_contracts = init_pair_contracts(factory_address, symbols, token_addresses)
    initial_reserves = get_reserves(pair_contracts, token_addresses, decimals)

    trade_conditions = TradeConditions(
        T,
        N,
        total_capital_denom,
        fee_multiplier,
        initial_alloc_proportion,
        target_alloc_proportion,
        initial_reserves,
    )
    return trade_conditions


@pytest.fixture
def small_trade_conditions():
    T = 2
    N = 4
    total_capital_denom = 1e6
    fee_multiplier = 0.997
    initial_alloc_proportion = np.array([0.1, 0.2, 0.3, 0.4])
    target_alloc_proportion = np.array([0.2, 0.2, 0.4, 0.2])
    initial_reserves = np.array(
        [
            [np.inf, 1e8, 5e6, 5e6],
            [1e8 / 1000, np.inf, 1e7 / 1000, np.inf],
            [5e6 / 50000, 1e7 / 50000, np.inf, np.inf],
            [5e6 / 20, np.inf, np.inf, np.inf],
        ]
    )

    # Check for no arbitrage
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if (
                        i == j
                        or j == k
                        or k == i
                        or initial_reserves[i, j] == np.inf
                        or initial_reserves[j, k] == np.inf
                        or initial_reserves[i, k] == np.inf
                ):
                    continue
                assert np.isclose(
                    initial_reserves[i][j]
                    / initial_reserves[j][i]
                    * initial_reserves[j][k]
                    / initial_reserves[k][j],
                    initial_reserves[i][k] / initial_reserves[k][i],
                    )

    trade_conditions = TradeConditions(
        T,
        N,
        total_capital_denom,
        fee_multiplier,
        initial_alloc_proportion,
        target_alloc_proportion,
        initial_reserves,
    )
    return trade_conditions


@pytest.fixture
def big_trade_conditions():
    rng = np.random.default_rng(0)

    T = 2
    N = 10
    total_capital_denom = 1e6
    fee_multiplier = 0.997
    initial_alloc_proportion = np.array(
        [0.3, 0.2, 0.2, 0.1, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01]
    )
    target_alloc_proportion = np.array([0.05, 0, 0.05, 0, 0.1, 0.2, 0.3, 0.3, 0, 0])
    prices = np.array([1, 1500, 50, 50, 50000, 0.05, 0.4, 40, 300, 400])

    initial_reserves_usd = np.full((N, N), np.inf)
    for i in range(N):
        for j in range(N):
            if i >= j:
                continue
            if i == 0 and j == 1:
                initial_reserves_usd[i][j] = 1e8
            elif i == 0:
                initial_reserves_usd[i][j] = rng.uniform(5e6, 8e6)
            elif i == 1:
                initial_reserves_usd[i][j] = rng.uniform(1e7, 3e7)
            else:
                initial_reserves_usd[i][j] = rng.uniform(1e6, 3e6)
    for i in range(N):
        for j in range(N):
            if i > j:
                initial_reserves_usd[i][j] = initial_reserves_usd[j][i]

    initial_reserves = np.array(
        [
            [
                initial_reserves_usd[reserve_asset][other_asset] / prices[reserve_asset]
                for other_asset in range(N)
            ]
            for reserve_asset in range(N)
        ]
    )

    indices_blocked = [
        [2, 3],
        [3, 4],
        [3, 6],
        [3, 8],
        [4, 5],
        [4, 7],
        [5, 9],
        [6, 7],
        [6, 8],
        [6, 9],
        [7, 8],
        [7, 9],
        [8, 9],
    ]
    for [i, j] in indices_blocked:
        initial_reserves[i][j] = initial_reserves[j][i] = np.inf

    # Check for no arbitrage, just as a sanity check
    # Usually there exists arbitrage opportunities in liquidity pools, but these cases will be covered in on-chain tests
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if (
                        i == j
                        or j == k
                        or k == i
                        or initial_reserves[i, j] == np.inf
                        or initial_reserves[j, k] == np.inf
                        or initial_reserves[i, k] == np.inf
                ):
                    continue
                assert np.isclose(
                    initial_reserves[i][j]
                    / initial_reserves[j][i]
                    * initial_reserves[j][k]
                    / initial_reserves[k][j],
                    initial_reserves[i][k] / initial_reserves[k][i],
                    )

    trade_conditions = TradeConditions(
        T,
        N,
        total_capital_denom,
        fee_multiplier,
        initial_alloc_proportion,
        target_alloc_proportion,
        initial_reserves,
    )
    return trade_conditions


@pytest.fixture
def random_weights_small(small_trade_conditions):
    return random_W0(
        small_trade_conditions.T,
        small_trade_conditions.N,
        small_trade_conditions.no_pair_indices,
        0,
    )


@pytest.fixture
def random_weights_big(big_trade_conditions):
    return random_W0(
        big_trade_conditions.T,
        big_trade_conditions.N,
        big_trade_conditions.no_pair_indices,
        0,
    )
