import "@nomiclabs/hardhat-ethers";
import "@nomiclabs/hardhat-waffle";
import { Contract, BigNumber } from "ethers";
import { formatEther } from "ethers/lib/utils";
import { expect } from "chai";


const AAVE_LTV_SCALE = 1e4;

export const calculateMaxBorrow = async (
  amountCollateral: BigNumber,
  lendingPool: Contract,
  oracle: Contract,
  collateralAssetAddress: string,
  borrowAssetAddress: string
) => {
  const reserveData = await lendingPool.getReserveData(borrowAssetAddress);
  const reserveConfiguration: BigNumber = reserveData.configuration[0];
  const maxLTV = reserveConfiguration.and((1 << 16) - 1); // x1e4

  const collateralAssetPrice = await oracle.getAssetPrice(
    collateralAssetAddress
  );
  const borrowAssetPrice = await oracle.getAssetPrice(borrowAssetAddress);

  const maxBorrow = maxLTV
    .mul(amountCollateral)
    .mul(collateralAssetPrice)
    .div(borrowAssetPrice.mul(AAVE_LTV_SCALE));

  // Check we haven't miscalculated
  expect(
    maxLTV
      .sub(
        maxBorrow
          .mul(AAVE_LTV_SCALE)
          .mul(borrowAssetPrice)
          .div(amountCollateral.mul(collateralAssetPrice))
      )
      .toNumber()
  ).to.be.lessThanOrEqual(
    1,
    "Miscalculated borrow amount from collateral and LTV"
  );

  return maxBorrow;
};


export const logBalances = async (
  address: string,
  assetContracts: Map<string, Contract>,
  header: string = ""
) => {
  console.log("\n----------" + header + "---------");
  for (let [key, contract] of assetContracts.entries()) {
    const balance = formatEther(
      await contract.balanceOf(address)
    );
    console.log(key + " balance: " + balance);
  }
  console.log("-------------------");
};
