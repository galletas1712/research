// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import { IFlashLoanReceiver } from "./IAAVEV2.sol";

interface IFlashLoanReceiverWithHook is IFlashLoanReceiver {
    function repayHook(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums
    ) external;
}
