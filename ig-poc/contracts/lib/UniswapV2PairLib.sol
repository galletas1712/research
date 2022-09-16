// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import "../interfaces/IERC20.sol";
import { IUniswapV2Pair } from "../interfaces/IUniswapV2.sol";

library UniswapV2PairLib {
    function getUnderlyingTokensAddresses(IUniswapV2Pair pair)
        public
        view
        returns (address tokenA, address tokenB)
    {
        tokenA = pair.token0();
        tokenB = pair.token1();
    }

    function getUnderlyingTokens(IUniswapV2Pair pair)
        public
        view
        returns (IERC20 tokenA, IERC20 tokenB)
    {
        tokenA = IERC20(pair.token0());
        tokenB = IERC20(pair.token1());
    }

    function amountUnderlyingTokens(IUniswapV2Pair pair, uint256 amountLP)
        public
        view
        returns (uint256 amountA, uint256 amountB)
    {
        uint256 totalSupply = pair.totalSupply();
        amountA = IERC20(pair.token0()).balanceOf(address(pair)) * amountLP / totalSupply;
        amountB = IERC20(pair.token1()).balanceOf(address(pair)) * amountLP / totalSupply;
    }
}
