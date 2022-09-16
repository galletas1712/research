# Rebalance optimizer

## Running Tests
- Run `npm install` and `pip install -r requirements.txt` in the root of the project directory to install dependencies
- Create a hardhat.config.js file in the root of the project directory if Brownie has not created one already with the following contents:
```
module.exports = {
  networks: {
    hardhat: {
      hardfork: "london",
      // base fee of 0 allows use of 0 gas price when testing
      initialBaseFeePerGas: 0,
      // brownie expects calls and transactions to throw on revert
      throwOnTransactionFailures: true,
      throwOnCallFailures: true,
      forking: {
        url: "https://eth-mainnet.alchemyapi.io/v2/YOU_API_KEY_HERE",
        blockNumber: 13546670,
      },
    },
  },
};
```
- Ensure `$PYTHONPATH` is set to the project root

**Note:** It is important that the network is forked from the exact block number indicated above, as liquidity is highly dynamic and the tests assume market conditions at the particular block height.
