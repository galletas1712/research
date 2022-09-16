import dotenv from "dotenv";
import "@nomiclabs/hardhat-waffle";
import { HardhatUserConfig } from "hardhat/config";

dotenv.config();

const config: HardhatUserConfig = {
  networks: {
    hardhat: {
      forking: {
        url: process.env.ALCHEMY_URL!,
        blockNumber: Number(process.env.BLOCK_NUMBER!),
      },
      allowUnlimitedContractSize: true,
    },
  },
  solidity: {
    compilers: [
      {
        version: "0.8.4",
      },
    ],
  },
  mocha: {
    timeout: 600000,
  },
};

export default config;
