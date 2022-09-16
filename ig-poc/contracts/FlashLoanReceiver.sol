// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import "hardhat/console.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

import "./interfaces/IERC20.sol";
import "./interfaces/IFlashLoanReceiverWithHook.sol";
import "./interfaces/IFlashLoanCallback.sol";
import { ILendingPoolAddressesProvider, ILendingPool } from "./interfaces/IAAVEV2.sol";

contract FlashLoanReceiver is IFlashLoanReceiverWithHook {

    ILendingPoolAddressesProvider public immutable ADDRESSES_PROVIDER;
    ILendingPool public immutable LENDING_POOL;
    address public immutable OWNER;

    constructor(ILendingPoolAddressesProvider provider, address owner) {
        ADDRESSES_PROVIDER = provider;
        LENDING_POOL = ILendingPool(provider.getLendingPool());
        OWNER = owner;
    }

    /**
        This function is called after your contract has received the flash loaned amount
     */
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        require(
            initiator == OWNER && msg.sender == address(LENDING_POOL),
            "Not authorized to execute flash loan"
        );
        for (uint256 i = 0; i < assets.length; i++) {
            IERC20(assets[i]).approve(OWNER, amounts[i]);
        }
        IFlashLoanCallback(OWNER).flashLoanCallback(assets, amounts, premiums, params);
        return true;
    }

    function repayHook(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums
    ) external override {
        require(msg.sender == OWNER, "Not authorized to repay flash loan");

        // At the end of your logic above, this contract owes
        // the flashloaned amounts + premiums.
        // Therefore ensure your contract has enough to repay
        // these amounts.

        // Approve the LendingPool contract allowance to *pull* the owed amount
        for (uint256 i = 0; i < assets.length; i++) {
            uint256 amountOwing = amounts[i] + premiums[i];
            IERC20(assets[i]).approve(address(LENDING_POOL), amountOwing);
        }
    }
}
