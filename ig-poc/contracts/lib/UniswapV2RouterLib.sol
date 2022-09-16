// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/math/Math.sol";

import "../lib/UniswapV2PairLib.sol";

import "../interfaces/IERC20.sol";
import { IUniswapV2Factory, IUniswapV2Router02 } from "../interfaces/IUniswapV2.sol";

library UniswapV2RouterLib {
    using UniswapV2PairLib for IUniswapV2Pair;

    function getPair(
        IUniswapV2Router02 router,
        address tokenA,
        address tokenB
    ) public view returns (address) {
        return IUniswapV2Factory(router.factory()).getPair(tokenA, tokenB);
    }

    function getPath(address tokenA, address tokenB) public pure returns (address[] memory path) {
        path = new address[](2);
        path[0] = tokenA;
        path[1] = tokenB;
    }

    function simpleSwapExactTokensForTokens(
        IUniswapV2Router02 router,
        address tokenA,
        address tokenB,
        uint256 amountA,
        address recipient,
        uint256 deadline
    ) public returns (bool) {
        address[] memory path = getPath(tokenA, tokenB);
        if (path[0] == path[1]) return false;
        (uint256 reserveA, uint256 reserveB, ) = IUniswapV2Pair(getPair(router, tokenA, tokenB))
        .getReserves();
        IERC20(tokenA).approve(address(router), amountA);
        router.swapExactTokensForTokens(
            amountA,
            router.getAmountOut(amountA, reserveA, reserveB),
            path,
            recipient,
            deadline
        );
        return true;
    }

    function simpleSwapTokensForExactTokens(
        IUniswapV2Router02 router,
        address tokenA,
        address tokenB,
        uint256 amountBOutput,
        address recipient,
        uint256 deadline
    ) public returns (bool) {
        address[] memory path = getPath(tokenA, tokenB);
        if (path[0] == path[1]) return false;
        (uint256 reserveA, uint256 reserveB, ) = IUniswapV2Pair(getPair(router, tokenA, tokenB))
        .getReserves();
        uint256 amountAMax = router.getAmountIn(amountBOutput, reserveA, reserveB);
        IERC20(tokenA).approve(address(router), amountAMax);
        router.swapTokensForExactTokens(amountBOutput, amountAMax, path, recipient, deadline);
        return true;
    }

    function splitAndRemoveLiquidity(
        IUniswapV2Router02 router,
        IUniswapV2Pair pair,
        uint256 amountLiquidity,
        uint256 deadline
    ) public {
        require(
            pair.balanceOf(address(this)) >= amountLiquidity,
            "Cannot split LP Token: balance too low"
        );
        (address tokenA, address tokenB) = pair.getUnderlyingTokensAddresses();
        (uint256 amountA, uint256 amountB) = pair.amountUnderlyingTokens(amountLiquidity);
        pair.approve(address(router), amountLiquidity);
        router.removeLiquidity(
            tokenA,
            tokenB,
            amountLiquidity,
            amountA, // should be exactly the same since no price slippage
            amountB,
            address(this),
            deadline
        );
    }

    function attemptRebalanceToDesiredRatio(
        IUniswapV2Router02 router,
        address tokenA,
        address tokenB,
        uint256 amountAMin,
        uint256 amountBMin,
        uint256 deadline
    ) public {
        IUniswapV2Pair pair = IUniswapV2Pair(
            IUniswapV2Factory(router.factory()).getPair(tokenA, tokenB)
        );
        uint256 balanceA = IERC20(tokenA).balanceOf(address(this));
        uint256 balanceB = IERC20(tokenB).balanceOf(address(this));
        // CAUTION: uses CREATE2 - might not work with local deployments of UniswapV2Pair
        (uint256 reserveA, uint256 reserveB, ) = pair.getReserves();

        if (balanceA < amountAMin && balanceB > amountBMin) {
            uint256 amountBAvailable = balanceB - amountBMin;
            uint256 amountBNeeded = router.getAmountIn(amountAMin - balanceA, reserveB, reserveA);
            simpleSwapExactTokensForTokens(
                router,
                tokenB,
                tokenA,
                Math.min(amountBAvailable, amountBNeeded),
                address(this),
                deadline
            );
        } else if (balanceB < amountBMin && balanceA > amountAMin) {
            uint256 amountAAvailable = balanceA - amountAMin;
            uint256 amountANeeded = router.getAmountIn(amountBMin - balanceB, reserveA, reserveB);
            simpleSwapExactTokensForTokens(
                router,
                tokenA,
                tokenB,
                Math.min(amountAAvailable, amountANeeded),
                address(this),
                deadline
            );
        }
    }
}
