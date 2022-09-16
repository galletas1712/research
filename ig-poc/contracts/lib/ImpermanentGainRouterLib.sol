// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";

import "../lib/UniswapV2RouterLib.sol";
import "../lib/UniswapV2PairLib.sol";

import "../interfaces/IERC20.sol";
import { ILendingPool } from "../interfaces/IAAVEV2.sol";
import { IUniswapV2Factory, IUniswapV2Pair, IUniswapV2Router02 } from "../interfaces/IUniswapV2.sol";

library ImpermanentGainRouterLib {
    using UniswapV2PairLib for IUniswapV2Pair;
    using UniswapV2RouterLib for IUniswapV2Router02;

    uint256 constant ROUNDING_ERROR_MARGIN = 1e5;
    uint256 constant ONE = 1e18;

    function getAmountDebt(ILendingPool lendingPool, address asset) public view returns (uint256) {
        return
            IERC20(lendingPool.getReserveData(asset).variableDebtTokenAddress).balanceOf(
                address(this)
            );
    }

    function getUnderlyingDebt(
        ILendingPool lendingPool,
        IUniswapV2Router02 router,
        address tokenA,
        address tokenB
    ) public view returns (uint256 amountA, uint256 amountB) {
        address pair = router.getPair(tokenA, tokenB);
        uint256 amountDebt = getAmountDebt(lendingPool, pair);
        (amountA, amountB) = IUniswapV2Pair(pair).amountUnderlyingTokens(amountDebt);
    }

    function depositCollateral(
        ILendingPool lendingPool,
        address asset,
        uint256 amount
    ) public {
        IERC20(asset).approve(address(lendingPool), amount);
        lendingPool.deposit(asset, amount, address(this), 0);
    }

    function initiateFlashLoan(
        ILendingPool lendingPool,
        address flashLoanReceiver,
        address assetA,
        address assetB,
        uint256 amountA,
        uint256 amountB,
        uint256 deadline
    ) public {
        address[] memory assets = new address[](2);
        assets[0] = assetA;
        assets[1] = assetB;

        uint256[] memory amounts = new uint256[](2);
        amounts[0] = amountA;
        amounts[1] = amountB;

        uint256[] memory modes = new uint256[](2);
        modes[0] = 0;
        modes[1] = 0;

        for (uint256 i = 0; i < assets.length; i++) {
            require(
                amounts[i] <
                    IERC20(assets[i]).balanceOf(
                        lendingPool.getReserveData(assets[i]).aTokenAddress
                    ),
                "Flash loan exceeds pool capacity"
            );
        }

        lendingPool.flashLoan(
            address(flashLoanReceiver),
            assets,
            amounts,
            modes,
            address(this),
            abi.encode(deadline),
            0
        );
    }

    function repayLendingPoolDebt(
        ILendingPool lendingPool,
        IUniswapV2Router02 router,
        address tokenA,
        address tokenB,
        uint256 amountA,
        uint256 amountB,
        uint256 deadline
    ) public {
        require(
            IERC20(tokenA).balanceOf(address(this)) == amountA &&
                IERC20(tokenB).balanceOf(address(this)) == amountB,
            "Amount to repay not consistent with contract balance"
        );

        // Restore liquidity
        IERC20(tokenA).approve(address(router), amountA);
        IERC20(tokenB).approve(address(router), amountB);
        router.addLiquidity(
            tokenA,
            tokenB,
            amountA,
            amountB,
            amountA * ONE / (ONE + ROUNDING_ERROR_MARGIN),
            amountB * ONE / (ONE + ROUNDING_ERROR_MARGIN),
            address(this),
            deadline
        );

        address pair = router.getPair(tokenA, tokenB);
        uint256 amountDebt = getAmountDebt(lendingPool, pair);
        require(
            IERC20(pair).balanceOf(address(this)) >= amountDebt,
            "Not enough balance to repay lending pool debt"
        );

        IERC20(pair).approve(address(lendingPool), amountDebt);
        lendingPool.repay(pair, amountDebt, 2, address(this));
    }

    function withdrawAllFromLendingPool(
        ILendingPool lendingPool,
        address tokenA,
        address tokenB,
        address collateralAsset
    ) public {
        lendingPool.withdraw(tokenA, type(uint256).max, address(this));
        lendingPool.withdraw(tokenB, type(uint256).max, address(this));
        if (collateralAsset != tokenA && collateralAsset != tokenB) {
            lendingPool.withdraw(collateralAsset, type(uint256).max, address(this));
        }
    }

    function ensureAssetBalanceRatio(
        IUniswapV2Router02 router,
        address tokenA,
        address tokenB,
        address collateralAsset,
        uint256 amountAMin,
        uint256 amountBMin,
        uint256 positionAmountCollateral,
        uint256 deadline
    ) internal {
        // Attempt to rebalance  without collateral
        // Adjust amountAMin and amountBMin to take into account position's initial collateral,
        // since we try to retain the owner's collateral asset balance if we can
        if (collateralAsset == tokenA) {
            amountAMin += positionAmountCollateral;
        } else if (collateralAsset == tokenB) {
            amountBMin += positionAmountCollateral;
        }
        router.attemptRebalanceToDesiredRatio(tokenA, tokenB, amountAMin, amountBMin, deadline);

        // Swap collateral if insufficient
        if (IERC20(tokenA).balanceOf(address(this)) < amountAMin) {
            router.simpleSwapTokensForExactTokens(
                collateralAsset,
                tokenA,
                amountAMin,
                address(this),
                deadline
            );
        }
        if (IERC20(tokenB).balanceOf(address(this)) < amountBMin) {
            router.simpleSwapTokensForExactTokens(
                collateralAsset,
                tokenB,
                amountBMin,
                address(this),
                deadline
            );
        }

        require(
            IERC20(tokenA).balanceOf(address(this)) >= amountAMin &&
                IERC20(tokenB).balanceOf(address(this)) >= amountBMin,
            "Could not close position: INTERNAL ERROR - insufficient funds"
        );
    }
}
