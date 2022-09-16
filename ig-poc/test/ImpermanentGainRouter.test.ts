import "@nomiclabs/hardhat-ethers";
import "@nomiclabs/hardhat-waffle";
import { ethers } from "hardhat";
import { Contract, Signer, BigNumber } from "ethers";
import { parseEther } from "ethers/lib/utils";
import { AddressZero } from "@ethersproject/constants";
import { expect } from "chai";

import ADDRESSES from "./resources/addresses/mainnet";

import IERC20_ABI from "./resources/abi/IERC20";
import UNISWAP_V2_ROUTER_ABI from "./resources/abi/UniswapV2Router02";

import {
  getAssetContracts,
  resetMainnetFork,
  fillWallet,
  setupAAVE,
  setupImpermanentGainRouter,
} from "./setup";
import { calculateMaxBorrow, logBalances } from "./util";

interface ITradeInfo {
  fromAssetSymbol: string;
  toAssetSymbol: string;
  amountFromAsset: number;
  traderWalletBalance: number[];
}

interface ITestParams {
  ownerWalletBalance: number[];
  collateralAssetSymbol: string;
  amountCollateral: number;
  borrowPairAddress: string;
  shouldTurnProfit: boolean;
  tradeInfo?: ITradeInfo;
}

const swapExactTokensForTokensToMe = async (
  me: Signer,
  fromAsset: string,
  toAsset: string,
  amountFromAsset: BigNumber
) => {
  const uniswapRouter = new Contract(
    ADDRESSES.UNISWAP_V2_ROUTER,
    UNISWAP_V2_ROUTER_ABI,
    ethers.provider
  );
  await uniswapRouter
    .connect(me)
    .swapExactTokensForTokens(
      amountFromAsset,
      0,
      [fromAsset, toAsset],
      await me.getAddress(),
      Date.now() + 60
    );
};

const openPosition = async (
  owner: Signer,
  amountCollateral: BigNumber,
  amountBorrow: BigNumber,
  collateralAssetAddress: string,
  borrowAssetAddress: string,
  impermanentGainRouterContract: Contract
) => {
  const collateralContract = new Contract(
    collateralAssetAddress,
    IERC20_ABI,
    ethers.provider
  );
  await collateralContract
    .connect(owner)
    .approve(impermanentGainRouterContract.address, amountCollateral);
  await impermanentGainRouterContract
    .connect(owner)
    .openPosition(
      collateralAssetAddress,
      borrowAssetAddress,
      amountCollateral,
      amountBorrow,
      Date.now() + 60
    );
};

const trade = async (
  assetFrom: Contract,
  assetTo: Contract,
  amountFromToTrade: BigNumber,
  trader: Signer
) => {
  await assetFrom
    .connect(trader)
    .approve(ADDRESSES.UNISWAP_V2_ROUTER, amountFromToTrade);
  await swapExactTokensForTokensToMe(
    trader,
    assetFrom.address,
    assetTo.address,
    amountFromToTrade
  );
};

describe("Tests", async () => {
  const symbols = ["WETH", "DAI", "WBTC"];

  // Signers
  let owner: Signer;
  let trader: Signer;
  let anybody: Signer;

  // Contracts
  let impermanentGainRouterContract: Contract;
  let assetContracts: Map<string, Contract>;

  // AAVE
  let lendingPool: Contract;
  let oracle: Contract;

  beforeEach(async () => {
    await resetMainnetFork();
    [owner, trader, anybody] = await ethers.getSigners();
    assetContracts = getAssetContracts(symbols);

    // Setup AAVE contracts
    [lendingPool, oracle] = await setupAAVE([ADDRESSES.DAI_WETH_PAIR]);
    impermanentGainRouterContract = await setupImpermanentGainRouter(owner);
  });

  it("Should have access control setup correctly", async () => {
    const NON_OWNER_ERR_MSG = "Ownable: caller is not the owner";
    await expect(
      impermanentGainRouterContract
        .connect(anybody)
        .setFlashLoanReceiver(AddressZero)
    ).to.be.revertedWith(NON_OWNER_ERR_MSG);
    await expect(
      impermanentGainRouterContract.flashLoanCallback(
        [AddressZero, AddressZero],
        [0, 0],
        [0, 0],
        ethers.utils.toUtf8Bytes("")
      )
    ).to.be.revertedWith("Unauthorized call to flash loan callback");
    await expect(
      impermanentGainRouterContract
        .connect(anybody)
        .openPosition(AddressZero, AddressZero, 0, 0, 0)
    ).to.be.revertedWith(NON_OWNER_ERR_MSG);
    await expect(
      impermanentGainRouterContract.connect(anybody).closePosition(0)
    ).to.be.revertedWith(NON_OWNER_ERR_MSG);
  });

  it("Collateral = 5 WBTC, ETH price in DAI plunges, debt repaid immediately (collateral unaffected)", async () => {
    await testWithParameters({
      ownerWalletBalance: [1, 1, 5e2],
      collateralAssetSymbol: "WBTC",
      amountCollateral: 5,
      borrowPairAddress: ADDRESSES.DAI_WETH_PAIR,
      shouldTurnProfit: true,
      tradeInfo: {
        fromAssetSymbol: "WETH",
        toAssetSymbol: "DAI",
        amountFromAsset: 1e6,
        traderWalletBalance: [1e12, 1, 2e10],
      },
    });
  });

  it("Collateral = 5 WBTC, ETH price in DAI moons, debt repaid immediately (collateral unaffected)", async () => {
    await testWithParameters({
      ownerWalletBalance: [1, 1, 5e2],
      collateralAssetSymbol: "WBTC",
      amountCollateral: 5,
      borrowPairAddress: ADDRESSES.DAI_WETH_PAIR,
      shouldTurnProfit: true,
      tradeInfo: {
        fromAssetSymbol: "DAI",
        toAssetSymbol: "WETH",
        amountFromAsset: 1e8,
        traderWalletBalance: [1, 1e13, 2e10],
      },
    });
  });

  it("Collateral = 1000 ETH, ETH price in DAI plunges, debt repaid immediately (collateral unaffected)", async () => {
    await testWithParameters({
      ownerWalletBalance: [2e3, 1, 1],
      collateralAssetSymbol: "WETH",
      amountCollateral: 1e3,
      borrowPairAddress: ADDRESSES.DAI_WETH_PAIR,
      shouldTurnProfit: true,
      tradeInfo: {
        fromAssetSymbol: "WETH",
        toAssetSymbol: "DAI",
        amountFromAsset: 1e6,
        traderWalletBalance: [1e12, 1, 2e10],
      },
    });
  });

  it("Collateral = 1000 ETH, ETH price in DAI moons, debt repaid immediately (collateral unaffected)", async () => {
    await testWithParameters({
      ownerWalletBalance: [2e3, 1, 1],
      collateralAssetSymbol: "WETH",
      amountCollateral: 1000,
      borrowPairAddress: ADDRESSES.DAI_WETH_PAIR,
      shouldTurnProfit: true,
      tradeInfo: {
        fromAssetSymbol: "DAI",
        toAssetSymbol: "WETH",
        amountFromAsset: 1e7,
        traderWalletBalance: [1, 1e13, 2e10],
      },
    });
  });

  const testWithParameters = async (testParams: ITestParams) => {
    await fillWallet(owner, symbols, testParams.ownerWalletBalance);

    await logBalances(
      await owner.getAddress(),
      assetContracts,
      "Owner before open position"
    );

    const collateralAssetAddress = assetContracts.get(
      testParams.collateralAssetSymbol
    )!.address;

    const amountCollateralFormatted = parseEther(
      testParams.amountCollateral.toString()
    );
    // Borrow with max LTV
    const amountLPBorrow = await calculateMaxBorrow(
      amountCollateralFormatted,
      lendingPool,
      oracle,
      collateralAssetAddress,
      testParams.borrowPairAddress
    );

    const initialBalances = await Promise.all(
      symbols.map(
        async (symbol) =>
          await assetContracts.get(symbol)!.balanceOf(await owner.getAddress())
      )
    );

    await openPosition(
      owner,
      amountCollateralFormatted,
      amountLPBorrow,
      collateralAssetAddress,
      testParams.borrowPairAddress,
      impermanentGainRouterContract
    );

    await expectContractBalanceZero("After open");

    // Opening another position should fail
    await expect(
      impermanentGainRouterContract
        .connect(owner)
        .openPosition(AddressZero, AddressZero, 0, 0, 0)
    ).to.be.revertedWith(
      "Unable to open position: another position already active"
    );

    // Whale trades (must be between assets in pair) to change reserves creating impermanent gain for position owner
    if (testParams.tradeInfo) {
      await fillWallet(
        trader,
        symbols,
        testParams.tradeInfo.traderWalletBalance
      );
      await trade(
        assetContracts.get(testParams.tradeInfo.fromAssetSymbol)!,
        assetContracts.get(testParams.tradeInfo.toAssetSymbol)!,
        parseEther(testParams.tradeInfo.amountFromAsset.toString()),
        trader
      );
    }

    await impermanentGainRouterContract
      .connect(owner)
      .closePosition(Date.now() + 60);

    await logBalances(
      await owner.getAddress(),
      assetContracts,
      "Owner after close position"
    );

    await expectContractBalanceZero("After close");

    if (testParams.shouldTurnProfit) {
      await Promise.all(
        symbols.map(async (symbol, i) =>
          expect(
            await assetContracts
              .get(symbol)!
              .balanceOf(await owner.getAddress())
          ).to.be.gte(initialBalances[i], "Position did not turn a profit")
        )
      );
    }

    // Close position after closed should fail
    await expect(
      impermanentGainRouterContract
        .connect(owner)
        .closePosition(Date.now() + 60)
    ).to.be.revertedWith("Unable to close position: no active position");
  };

  const expectContractBalanceZero = (stage: string) =>
    Promise.all(
      symbols.map(async (asset) => {
        const assetContract = assetContracts.get(asset)!;
        expect(
          await assetContract.balanceOf(impermanentGainRouterContract.address)
        ).to.equal(0, stage + ": " + "balance of " + asset + " non-zero");
      })
    );
});

// Parameters
// 1. Collateral asset
// 2. Interest eats into collateral (enumerate all cases in _ensureFlashLoanRepayable and _attemptRebalanceWithoutCollateral)
// 3. Price relative to DAI (or if price even moves at all)
