// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

import "./lib/UniswapV2RouterLib.sol";
import "./lib/UniswapV2PairLib.sol";
import "./lib/ImpermanentGainRouterLib.sol";

import "./interfaces/IERC20.sol";
import "./interfaces/IFlashLoanReceiverWithHook.sol";
import "./interfaces/IFlashLoanCallback.sol";
import { ILendingPoolAddressesProvider, ILendingPool } from "./interfaces/IAAVEV2.sol";
import { IUniswapV2Pair, IUniswapV2Router02 } from "./interfaces/IUniswapV2.sol";

/// @title Impermanent Gain Router
/// @author Atto
/// @notice Used to open/close short positions against impermanent loss
/// @notice Can only be used by the deployer of the contract
contract ImpermanentGainRouter is Ownable, IFlashLoanCallback {
    using UniswapV2PairLib for IUniswapV2Pair;
    using UniswapV2RouterLib for IUniswapV2Router02;

    uint256 constant ROUNDING_ERROR_MARGIN = 1e5;
    uint256 constant ONE = 1e18;

    // Periphery
    ILendingPoolAddressesProvider internal LENDING_POOL_ADDRESSES_PROVIDER;
    ILendingPool internal immutable LENDING_POOL;
    IUniswapV2Router02 internal immutable ROUTER;
    IFlashLoanReceiverWithHook internal flashLoanReceiver;

    // State
    struct Position {
        bool active;
        uint256 amountCollateral;
        address collateralAsset;
        address tokenA;
        address tokenB;
    }
    Position internal currPosition;

    modifier onlyFlashLoanReceiver() {
        require(
            msg.sender == address(flashLoanReceiver),
            "Unauthorized call to flash loan callback"
        );
        _;
    }

    constructor(address lendingPoolAddressesProvider_, address router_) {
        LENDING_POOL_ADDRESSES_PROVIDER = ILendingPoolAddressesProvider(
            lendingPoolAddressesProvider_
        );
        LENDING_POOL = ILendingPool(LENDING_POOL_ADDRESSES_PROVIDER.getLendingPool());
        ROUTER = IUniswapV2Router02(router_);
    }

    /// @dev Sets flash loan receiver for this contract to allow calling flashLoanCallback
    /// @param _flashLoanReceiver Address of flash loan receiver
    function setFlashLoanReceiver(address _flashLoanReceiver) public onlyOwner {
        flashLoanReceiver = IFlashLoanReceiverWithHook(_flashLoanReceiver);
    }

    /// @notice Open a short position against impermanent loss
    /// @notice Only one position can be opened at a time
    /// @dev This contract should not hold any assets after transaction is complete
    /// @dev Collateral must be approved before call so that the contact can pull the collateral in
    /// @dev Collateral asset can be a constituent of the borrowed pair
    /// @param collateralAsset ERC20 token address of collateral asset
    /// @param borrowPair Address of pair to borrow for shorting (must adhere to IUniswapV2Pair interface, could be Uniswap V2, Sushiswap, etc)
    /// @param amountCollateral Amount of collateral asset to deposit
    /// @param amountBorrow Number of shares of LP positions to borrow
    /// @param deadline Deadline for executing ERC20 swaps
    function openPosition(
        address collateralAsset,
        address borrowPair,
        uint256 amountCollateral,
        uint256 amountBorrow,
        uint256 deadline
    ) external onlyOwner {
        require(!currPosition.active, "Unable to open position: another position already active");
        require(
            IERC20(collateralAsset).allowance(msg.sender, address(this)) >= amountCollateral,
            "Could not create new position: not enough allowance"
        );

        IERC20(collateralAsset).transferFrom(msg.sender, address(this), amountCollateral);

        // Deposit, borrow, split
        ImpermanentGainRouterLib.depositCollateral(
            LENDING_POOL,
            collateralAsset,
            amountCollateral
        );
        LENDING_POOL.borrow(borrowPair, amountBorrow, 2, 0, address(this));
        ROUTER.splitAndRemoveLiquidity(IUniswapV2Pair(borrowPair), amountBorrow, deadline);

        // Redeposit splitted collateral
        (address tokenA, address tokenB) = IUniswapV2Pair(borrowPair).getUnderlyingTokensAddresses();
        ImpermanentGainRouterLib.depositCollateral(
            LENDING_POOL,
            tokenA,
            IERC20(tokenA).balanceOf(address(this))
        );
        ImpermanentGainRouterLib.depositCollateral(
            LENDING_POOL,
            tokenB,
            IERC20(tokenB).balanceOf(address(this))
        );

        currPosition = Position(
            true,
            amountCollateral,
            collateralAsset,
            tokenA,
            tokenB
        );
    }

    /// @notice Closes the current position, repaying back all debts and realizing profits
    /// @notice Existing position must have been opened
    /// @dev Uses AAVE flash loan to ensure withdrawal is possible without getting liquidated
    /// @param deadline Deadline for executing ERC20 swaps
    function closePosition(uint256 deadline) external onlyOwner {
        require(currPosition.active, "Unable to close position: no active position");
        require(
            address(flashLoanReceiver) != address(0),
            "Unable to close position: flash loan receiver not set"
        );

        (uint256 amountAMin, uint256 amountBMin) = ImpermanentGainRouterLib.getUnderlyingDebt(
            LENDING_POOL,
            ROUTER,
            currPosition.tokenA,
            currPosition.tokenB
        );
        // Initiate flash loan and continue closing position in flashLoanCallback
        // Need to borrow slightly more to cover rounding error from getting underlying token amounts
        ImpermanentGainRouterLib.initiateFlashLoan(
            LENDING_POOL,
            address(flashLoanReceiver),
            currPosition.tokenA,
            currPosition.tokenB,
            amountAMin + amountAMin * ROUNDING_ERROR_MARGIN / ONE,
            amountBMin + amountBMin * ROUNDING_ERROR_MARGIN / ONE,
            deadline
        );
    }

    /// @dev Callback function for flash loan receiver to call after receiving borrowed funds
    /// @dev Only callable by flash loan receiver
    /// @param assets Addresses of borrowed assets. Should be the constituent tokens of borrowed pair
    /// @param amounts Amounts of borrowed assets
    /// @param premiums Cost of borrowing each asset
    /// @param params Parameters passed back from LendingPool contract. Should be byte-encoded deadline
    function flashLoanCallback(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        bytes calldata params
    ) external override onlyFlashLoanReceiver {
        require(
            assets[0] == currPosition.tokenA && assets[1] == currPosition.tokenB,
            "Wrong assets from flash loan"
        );
        require(
            assets.length == 2 && amounts.length == 2 && premiums.length == 2,
            "Wrong number of assets from flash loan"
        );

        // Transfer assets from flash loan receiver to this contract
        for (uint256 i = 0; i < assets.length; i++) {
            IERC20(assets[i]).transferFrom(address(flashLoanReceiver), address(this), amounts[i]);
        }

        uint256 deadline = abi.decode(params, (uint256));

        // Finish closing position
        ImpermanentGainRouterLib.repayLendingPoolDebt(
            LENDING_POOL,
            ROUTER,
            currPosition.tokenA,
            currPosition.tokenB,
            amounts[0],
            amounts[1],
            deadline
        );

        require(
            ImpermanentGainRouterLib.getAmountDebt(
                LENDING_POOL,
                ROUTER.getPair(currPosition.tokenA, currPosition.tokenB)
            ) == 0,
            "Close position internal error: not all lending pool debts repaid"
        );

        ImpermanentGainRouterLib.withdrawAllFromLendingPool(
            LENDING_POOL,
            currPosition.tokenA,
            currPosition.tokenB,
            currPosition.collateralAsset
        );

        ImpermanentGainRouterLib.ensureAssetBalanceRatio(
            ROUTER,
            currPosition.tokenA,
            currPosition.tokenB,
            currPosition.collateralAsset,
            amounts[0] + premiums[0],
            amounts[1] + premiums[1],
            currPosition.amountCollateral,
            deadline
        );

        // Repay flash loan
        IERC20(assets[0]).transfer(address(flashLoanReceiver), amounts[0] + premiums[0]);
        IERC20(assets[1]).transfer(address(flashLoanReceiver), amounts[1] + premiums[1]);
        flashLoanReceiver.repayHook(assets, amounts, premiums);

        // Transfer funds out of contract
        IERC20(currPosition.collateralAsset).transfer(
            owner(),
            IERC20(currPosition.collateralAsset).balanceOf(address(this))
        );
        IERC20(currPosition.tokenA).transfer(
            owner(),
            IERC20(currPosition.tokenA).balanceOf(address(this))
        );
        IERC20(currPosition.tokenB).transfer(
            owner(),
            IERC20(currPosition.tokenB).balanceOf(address(this))
        );

        // Reset position active flag
        currPosition.active = false;
    }
}
