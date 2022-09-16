import numpy as np
import pytest

from brownie import accounts, Contract
from optimizer.adapter import read_abi, init_network, get_factory_contract, init_token_contracts, get_decimals, \
    init_pair_contracts, get_reserves, gen_trades
from optimizer.circuit import next_iter
from optimizer.core import no_back_forth_cons


@pytest.fixture
def module_contracts(factory_address, symbols, token_addresses):
    init_network()
    factory_contract = get_factory_contract(factory_address)
    token_contracts = init_token_contracts(symbols, token_addresses)
    pair_contracts = init_pair_contracts(factory_address, symbols, token_addresses)
    return factory_contract, token_contracts, pair_contracts


@pytest.fixture
def pseudo_faucets():
    return [
        None,
        accounts.at("0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8", force=True),  # USDC
        accounts.at("0x4cae362d7f227e3d306f70ce4878e245563f3069", force=True),  # BOND
        accounts.at("0xd632f22692fac7611d2aa1c0d552930d43caed3b", force=True),  # FRAX
    ]


@pytest.fixture
def router_address():
    return "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"


@pytest.fixture
def router_contract(router_address):
    return Contract.from_abi(
        "UniswapV2Router", router_address, read_abi("UniswapV2Router.json")
    )


@pytest.fixture
def initial_alloc():
    # All 1e5 dollars
    return np.array([25, 1e5, 2941, 1e5])


@pytest.fixture
def weights():
    return np.array(
        [
            [0.3, 0.8, 0.0, 0.0],
            [0.0, 0.2, 0.7, 0.4],
            [0.7, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.6],  # no pair indices
        ]
    )


def test_next_iter_matches_uniswap(
        symbols,
        token_addresses,
        module_contracts,
        pseudo_faucets,
        router_contract,
        weights,
        initial_alloc,
):
    # If back and forth exists, then test will fail since reserves get mutated
    assert (no_back_forth_cons(1, weights.shape[0], weights.reshape(-1)) == 0).all()

    factory_contract, token_contracts, pair_contracts = module_contracts
    decimals = np.array(get_decimals(token_contracts))
    portfolio_account = accounts[0]

    N = len(initial_alloc)
    initial_alloc_wei = initial_alloc * (10 ** decimals)
    initial_reserves = get_reserves(pair_contracts, token_addresses, decimals)

    print(f"Portfolio account ETH balance: {portfolio_account.balance() / (10 ** 18)}")
    for i in range(1, N):
        print(
            f"Pseudo-faucet balance of {symbols[i]}: {token_contracts[i].balanceOf(pseudo_faucets[i].address) / (10 ** decimals[i])}"
        )

    # Special case for WETH
    # pseudo_faucet.transfer(portfolio_account, initial_alloc_wei[0])
    token_contracts[0].deposit(
        {"from": portfolio_account, "amount": initial_alloc_wei[[0]]}
    )
    for i in range(1, N):
        token_contracts[i].transfer(
            portfolio_account, initial_alloc_wei[i], {"from": pseudo_faucets[i]}
        )

    trades = gen_trades(weights, initial_alloc_wei)
    for trade in trades:
        print("Trade:", symbols[trade.src], symbols[trade.dest], trade.amount)
        print(
            "Allowance:",
            token_contracts[trade.src].allowance(
                portfolio_account.address, router_contract.address
            ),
        )
        print(
            "Balance  :",
            token_contracts[trade.src].balanceOf(portfolio_account.address),
        )
        print("Pair address:", pair_contracts[trade.src][trade.dest].address)

        token_contracts[trade.src].approve(
            router_contract.address, trade.amount, {"from": portfolio_account}
        )
        router_contract.swapExactTokensForTokens(
            trade.amount,
            0,
            [token_contracts[trade.src].address, token_contracts[trade.dest].address],
            portfolio_account.address,
            2e9,  # really far deadline
            {"from": portfolio_account},
        )

    amounts_out_from_on_chain = np.array(
        [
            token_contracts[i].balanceOf(portfolio_account.address)
            / (10 ** decimals[i])
            for i in range(N)
        ]
    )

    amounts_out_model, reserves_after_model = next_iter(
        weights, initial_alloc, initial_reserves, 0.997
    )
    print(amounts_out_model)
    print(amounts_out_from_on_chain)
    assert np.isclose(amounts_out_from_on_chain, amounts_out_model).all()

    reserves_after_on_chain = get_reserves(pair_contracts, token_addresses, decimals)
    assert np.isclose(reserves_after_on_chain, reserves_after_model).all()
