// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

interface IFlashLoanCallback {
    function flashLoanCallback(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        bytes calldata params
    ) external;
}
