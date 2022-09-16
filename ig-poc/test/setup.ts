import dotenv from "dotenv";
import "@nomiclabs/hardhat-ethers";
import { ethers, network } from "hardhat";
import { Contract, Signer, BigNumber } from "ethers";
import { parseEther } from "ethers/lib/utils";

import ADDRESSES from "./resources/addresses/mainnet";
import WETH_ABI from "./resources/abi/WETH";
import AAVE_ORACLE_ABI from "./resources/abi/PriceOracle";
import AAVE_LENDING_POOL_CONFIGURATOR_ABI from "./resources/abi/LendingPoolConfigurator";
import IERC20_ABI from "./resources/abi/IERC20";

import { assert } from "console";

dotenv.config();

const formatAddressBalance = (amount: BigNumber): string | undefined => {
  assert(!amount.eq(0));
  const trimmed = amount.toHexString().slice(3);
  for (let i = 0; i < trimmed.length; i++) {
    if (trimmed[i] != "0") {
      return "0x" + trimmed.slice(i);
    }
  }
};

const setAddressBalance = async (address: string, amount: BigNumber) =>
  await network.provider.send("hardhat_setBalance", [
    address,
    formatAddressBalance(amount),
  ]);

export const getAssetContracts = (assets: string[]) => {
  let assetContracts: Map<string, Contract> = new Map();
  assets.forEach((asset) => {
    let contract =
      asset == "WETH"
        ? new Contract(ADDRESSES.WETH, WETH_ABI, ethers.provider)
        : new Contract(
            (ADDRESSES.ASSETS as any)[asset].contractAddress,
            IERC20_ABI,
            ethers.provider
          );
    assetContracts.set(asset, contract);
  });
  return assetContracts;
};

export const resetMainnetFork = async () => {
  await network.provider.request({
    method: "hardhat_reset",
    params: [
      {
        forking: {
          jsonRpcUrl: process.env.ALCHEMY_URL!,
          blockNumber: Number(process.env.BLOCK_NUMBER!),
        },
      },
    ],
  });
};

export const fillWallet = async (
  wallet: Signer,
  assets: string[],
  amounts: number[]
) => {
  const GAS = parseEther("1000000000000000000");
  assert(assets.length == amounts.length);

  const amountsBigNum = amounts.map((x) => parseEther(x.toString()));
  const assetContracts = getAssetContracts(assets);

  for (let i = 0; i < assets.length; i++) {
    const assetContract: Contract = assetContracts.get(assets[i])!;

    await setAddressBalance(await wallet.getAddress(), GAS);

    if (assets[i] == "WETH") {
      await setAddressBalance(
        await wallet.getAddress(),
        amountsBigNum[i].add(GAS)
      );
      assetContract.connect(wallet).deposit({ value: amountsBigNum[i] });
    } else {
      const obj = (ADDRESSES.ASSETS as any)[assets[i]];

      await network.provider.request({
        method: "hardhat_impersonateAccount",
        params: [obj.minter],
      });

      const signer = ethers.provider.getSigner(obj.minter);
      await setAddressBalance(obj.minter, GAS);
      await assetContract
        .connect(signer)
        .mint(await wallet.getAddress(), amountsBigNum[i]);

      await network.provider.request({
        method: "hardhat_stopImpersonatingAccount",
        params: [obj.minter],
      });
    }
  }
};

export const setupAAVE = async (borrowReservesToEnable: string[]) => {
  // Attach contracts
  const addressesProvider = await ethers.getContractAt(
    "ILendingPoolAddressesProvider",
    ADDRESSES.AAVE_AMM_MARKET_ADDRESSES_PROVIDER
  );
  const lendingPool = await ethers.getContractAt(
    "ILendingPool",
    await addressesProvider.getLendingPool()
  );
  const oracle = new Contract(ADDRESSES.AAVE_ORACLE, AAVE_ORACLE_ABI, ethers.provider);
  const poolAdminAddress = await addressesProvider.getPoolAdmin();
  const configuratorAddress =
    await addressesProvider.getLendingPoolConfigurator();
  const poolAdminSigner = ethers.provider.getSigner(poolAdminAddress);
  const configuratorContract = new Contract(
    configuratorAddress,
    AAVE_LENDING_POOL_CONFIGURATOR_ABI,
    ethers.provider
  );

  await network.provider.send("hardhat_setBalance", [
    poolAdminAddress,
    parseEther("10").toHexString(),
  ]);

  // Enable variable rate borrow on reserve
  await network.provider.request({
    method: "hardhat_impersonateAccount",
    params: [poolAdminAddress],
  });
  for (let reserveAddress of borrowReservesToEnable) {
    await configuratorContract
      .connect(poolAdminSigner)
      .enableBorrowingOnReserve(reserveAddress, false);
  }
  await network.provider.request({
    method: "hardhat_stopImpersonatingAccount",
    params: [poolAdminAddress],
  });

  return [lendingPool, oracle];
};

export const setupImpermanentGainRouter = async (owner: Signer) => {
  // Deploy required libraries
  const uniswapV2PairLib = await ethers
    .getContractFactory("UniswapV2PairLib")
    .then((factory) => factory.deploy());
  const uniswapV2RouterLib = await ethers
    .getContractFactory("UniswapV2RouterLib", {
      libraries: {
        UniswapV2PairLib: uniswapV2PairLib.address,
      },
    })
    .then((factory) => factory.deploy());
  const impermanentGainRouterLib = await ethers
    .getContractFactory("ImpermanentGainRouterLib", {
      libraries: {
        UniswapV2PairLib: uniswapV2PairLib.address,
        UniswapV2RouterLib: uniswapV2RouterLib.address,
      },
    })
    .then((factory) => factory.deploy());

  // Deploy ImpermanentGainRouter contract
  const impermanentGainRouterContract = await (
    await ethers.getContractFactory("ImpermanentGainRouter", {
      libraries: {
        UniswapV2PairLib: uniswapV2PairLib.address,
        UniswapV2RouterLib: uniswapV2RouterLib.address,
        ImpermanentGainRouterLib: impermanentGainRouterLib.address,
      },
    })
  )
    .connect(owner)
    .deploy(ADDRESSES.AAVE_AMM_MARKET_ADDRESSES_PROVIDER, ADDRESSES.UNISWAP_V2_ROUTER);

  // Setup flash loan receiver
  const flashLoanReceiverContract = await (
    await ethers.getContractFactory("FlashLoanReceiver")
  )
    .connect(owner)
    .deploy(
      ADDRESSES.AAVE_AMM_MARKET_ADDRESSES_PROVIDER,
      impermanentGainRouterContract.address
    );

  impermanentGainRouterContract
    .connect(owner)
    .setFlashLoanReceiver(flashLoanReceiverContract.address);

  return impermanentGainRouterContract;
};
