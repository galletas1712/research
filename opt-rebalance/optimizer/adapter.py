import json
import math
import os
from typing import NamedTuple, List

import numpy as np
from brownie import network, Contract


class Trade(NamedTuple):
    src: int
    dest: int
    amount: float


def gen_trades(W: np.ndarray, initial_alloc: np.ndarray) -> List[Trade]:
    """Generates trades for one iteration, given weights and initial allocation."""
    assert len(W.shape) == 2 and W.shape[0] == W.shape[1]
    N = W.shape[0]
    assert initial_alloc.shape == (N,)

    trades = []
    amounts_to_trade = W * initial_alloc
    for i in range(amounts_to_trade.shape[0]):
        for j in range(amounts_to_trade.shape[1]):
            if i == j or math.isclose(amounts_to_trade[i, j], 0):
                continue
            trades.append(Trade(j, i, amounts_to_trade[i, j]))

    return trades


def read_abi(filename: str) -> List[any]:
    """Reads a select file as abi from abi directory."""
    abi_path_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "abi")
    factory_abi_path = os.path.join(abi_path_root, filename)
    f = open(factory_abi_path)
    result = json.load(f)
    f.close()
    return result


def init_network():
    """Initializes a new hardhat fork network, resetting it if an instance already exists."""
    if network.is_connected():
        network.disconnect()
    network.connect("hardhat")


def get_factory_contract(factory_address: str) -> Contract:
    """Instantiates a factory contract on local mainnet fork from abi."""
    return Contract.from_abi(
        "UniswapV2Factory", factory_address, read_abi("UniswapV2Factory.json")
    )


def init_token_contracts(symbols: List[str], token_addresses: List[str]) -> List[Contract]:
    """Instantiates token contracts needed to simulate on-chain interactions on local mainnet fork from abi."""
    assert len(symbols) == len(token_addresses)
    token_contracts = []
    for symbol, token_address in zip(symbols, token_addresses):
        if symbol in ["ETH", "WETH"]:
            token_contracts.append(
                Contract.from_abi(symbol, token_address, read_abi("WETH.json"))
            )
        else:
            token_contracts.append(
                Contract.from_abi(symbol, token_address, read_abi("ERC20.json"))
            )
    return token_contracts


def init_pair_contracts(
    factory_address: str, symbols: List[str], token_addresses: List[str]
) -> List[List[Contract]]:
    """Instanties contracts for each AMM pair on local mainnet fork from abi."""
    assert len(symbols) == len(token_addresses)
    factory_contract = get_factory_contract(factory_address)
    pair_contracts = [
        [None for _ in range(len(token_addresses))] for _ in range(len(token_addresses))
    ]
    for i in range(len(token_addresses)):
        for j in range(i + 1, len(token_addresses)):
            address = factory_contract.getPair(token_addresses[i], token_addresses[j])
            if address != "0x0000000000000000000000000000000000000000":
                pair_contracts[i][j] = pair_contracts[j][i] = Contract.from_abi(
                    f"{symbols[i]}-{symbols[j]} Pair",
                    address,
                    read_abi("UniswapV2Pair.json"),
                )
    return pair_contracts


def get_decimals(token_contracts: List[Contract]) -> List[int]:
    """Reads decimals for each token from token contract instances on local mainnet fork."""
    return [token_contract.decimals() for token_contract in token_contracts]


def get_reserves(
    pair_contracts: List[List[Contract]],
    token_addresses: List[str],
    decimals: List[int],
) -> np.ndarray:
    """Reads reserve data for each pair from LP pair contract instances on local mainnet fork."""
    N = len(pair_contracts)
    reserves = [[None for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if pair_contracts[i][j] is not None:
                reserve0, reserve1, _ = pair_contracts[i][j].getReserves()
                if (
                    str(pair_contracts[i][j].token0()).lower()
                    == token_addresses[i].lower()
                ):
                    reserves[i][j] = reserve0 / (10 ** decimals[i])
                elif (
                    str(pair_contracts[i][j].token1()).lower()
                    == token_addresses[i].lower()
                ):
                    reserves[i][j] = reserve1 / (10 ** decimals[i])
                else:
                    assert False, "Token not in pair"
            else:
                reserves[i][j] = np.inf
    return np.array(reserves)
