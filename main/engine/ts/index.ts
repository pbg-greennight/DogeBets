// index.ts

import chalk from 'chalk';
import { BigNumber, ethers } from 'ethers';
import { claimer } from './claimer';
import { dogeBetsPrediction, wallet, sleep, formatTime, GLOBAL_CONFIG, placeBet,
         getCurrentRoundDetails, publishEpochToFlask, getRoundData, executeBetBasedOnTrend} from './lib';
import { DogeBetsPrediction, DogeBetsPrediction__factory } from './types/ethers-contracts';
import { saveRoundData, PreRoundData } from './record';
const { redBright: brightRed, greenBright: brightGreen, yellowBright: brightYellow, blueBright: brightBlue,
        magentaBright: brightMagenta, cyanBright:   brightCyan, whiteBright:  brightWhite, white, gray } = chalk;

///////////////////////////////////////////  Bot Activation  ///////////////////////////////////////////////////////////
console.log(brightBlue("DogeBets Predictions Bot Starting. Amount to Bet:", GLOBAL_CONFIG.Wager, "BNB."));
console.log(brightBlue(`------------------------------------------------------------------------   Active Bot   ------------------------------------------------------------------------------------------`));
/////////////////////////////////////////  Main Logic  /////////////////////////////////////////////////////////////////
async function mainLogic() {
  // initial claimer with retry
  while (true) {
    try {
      await claimer(dogeBetsPrediction, wallet);
      break;
    } catch (err: any) {
      console.error(brightRed(`Initial claimer failed: ${err.message}. Retrying in 5s…`));
      await sleep(5000);
    }
  }
  while (true) {
    try {
      const { epoch, startTimestamp, lockTimestamp, closeTimestamp,
              lockPrice, closePrice } = await getCurrentRoundDetails();
      await publishEpochToFlask(epoch.toNumber() + 1, startTimestamp);
      const roundData = await getRoundData(epoch);
      const currentTime = formatTime(Date.now() / 1000);
      const sleepDuration = roundData.RoundTime_5m;
      // record pre-round
      PreRoundData(currentTime, roundData.nextEpochTime, roundData.epoch.toNumber());
      console.log(
        white(`**** Pre ${roundData.epoch.toNumber()} Round ****  `) +
        white(`Time: ${currentTime}, `) +
        white(`Current Epoch ${roundData.epoch.toNumber() - 1}  |  Next Epoch ${roundData.epoch.toNumber()} at ${roundData.nextEpochTime}`)
      );
      // wait until lock
      await sleep(sleepDuration * 1000);
      // begin round
      console.log(
        white(`**** Begin ${roundData.epoch} Round ****  `) +
        white(`Time: ${formatTime(Date.now() / 1000)}, `) +
        white(`Previous Round: ${roundData.previousEpoch} `) +
        white(`Start Price: ${roundData.RoundLockPrice}, `) +
        white(`End Price: ${roundData.RoundClosePrice}, `) +
        white(`Price Diff: ${roundData.priceDifference}, `) +
        white(`Next Epoch ${roundData.epoch} at ${roundData.nextEpochTime}`)
      );
      // persist outcome
      saveRoundData( currentTime, roundData.previousEpoch.toNumber(), roundData.RoundLockPrice, roundData.RoundClosePrice,
                     roundData.priceDifference, roundData.epoch.toNumber(), roundData.nextEpochTime );
      // execute bet
      await executeBetBasedOnTrend(epoch.toNumber());
      console.log(gray(`---------------------------------------------------------------------------------------------------------------------------------------------------------------------------`));
      // post-round claim & pause
      await claimer(dogeBetsPrediction, wallet);
      await sleep(20000);
      console.log(white(`**** Post ${roundData.epoch.toNumber()} Round ****  `) +
                  white(`Time: ${formatTime(Date.now() / 1000)}, `) +
                  white(`Current Epoch ${roundData.epoch.toNumber() - 1}  |  Next Epoch ${roundData.epoch.toNumber()} at ${roundData.nextEpochTime}`));
      console.log(gray(`///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////`));
    } catch (err: any) {
//      console.error(brightYellow(`⚠️  Caught error: ${err.message}. Retrying in 10s…`));
      console.error(brightYellow(`⚠️  Caught error: Retrying in 10s…`));
      await sleep(10000);
    }
  }
}
mainLogic().catch(err => {
  console.error(brightRed(`🔥 Fatal error: ${err.stack || err}`));
  process.exit(1);
});
