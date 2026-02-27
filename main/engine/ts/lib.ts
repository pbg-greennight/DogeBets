//  lib.ts

import { BigNumber, ethers } from 'ethers';
import { formatEther, parseEther } from "@ethersproject/units";
import { Wallet } from "@ethersproject/wallet";
import { DogeBetsPrediction, DogeBetsPrediction__factory } from './types/ethers-contracts';
import dotenv from 'dotenv';
import { clear } from "console";
import chalk from 'chalk';
import moment from 'moment-timezone';
import * as fs from 'fs';
import * as path from 'path';
import { red, green, blue, yellow, magenta, cyan, white, gray, black, bgRed, bgGreen, bgBlue, bgYellow, bgMagenta, bgCyan, bgWhite, bgGray, bgBlack } from 'chalk';
const brightRed = chalk.redBright; const brightGreen = chalk.greenBright; const brightYellow = chalk.yellowBright; const brightBlue = chalk.blueBright;
const brightMagenta = chalk.magentaBright; const brightCyan = chalk.cyanBright; const brightWhite = chalk.whiteBright;

dotenv.config();

const TREND_LOG_FILE = path.join(__dirname, 'json/DB_rounds_trend.json');

export const GLOBAL_CONFIG = {
  DogeBets_ADDRESS: "0xAfA97b96325AA5595061018f6AA3F64C612a83d9",
  Wager: process.env.BET_AMOUNT || "0.0975",
  BSC_RPC: "https://bsc-dataseed.binance.org/",
  PRIVATE_KEY: process.env.PRIVATE_KEY,
  WAITING_TIME: 9,
};

const wager = GLOBAL_CONFIG.Wager || '0.0975';
clear();
if (!GLOBAL_CONFIG.PRIVATE_KEY) {
  console.log(brightRed("❌ Private key not found in .env. Please add your private key and restart."));
  process.exit(1);
}

export const provider = new ethers.providers.JsonRpcProvider(GLOBAL_CONFIG.BSC_RPC);
export const wallet = new ethers.Wallet(GLOBAL_CONFIG.PRIVATE_KEY, provider);
export const dogeBetsPrediction = DogeBetsPrediction__factory.connect(GLOBAL_CONFIG.DogeBets_ADDRESS, wallet);

export const getRoundData = async (epoch: BigNumber) => {
  const currentTimeInSeconds = Math.floor(Date.now() / 1000);
  const { startTimestamp, lockTimestamp, closeTimestamp, lockPrice, closePrice } = await getCurrentRoundDetails();

  const roundStartTimeInSeconds = BigNumber.from(startTimestamp).toNumber();
  const currentEpochStartLockTime = formatTime(startTimestamp);
  const NextEpochStartLockTime = formatTime(lockTimestamp);
  const currentEpochCloseEndTime = formatTime(closeTimestamp);
  const nextEpochStartLockTimeInSeconds = parseInt(NextEpochStartLockTime, 10);
  const RoundTime_5m = Math.abs((currentTimeInSeconds - lockTimestamp) + GLOBAL_CONFIG.WAITING_TIME);
  const nextEpochTime = moment.unix(lockTimestamp).tz("America/New_York").format("hh:mm:ss A");

  const previousEpoch = BigNumber.from(epoch).sub(2);
  const lastEpoch = BigNumber.from(epoch).sub(1);
  const previousRound = await dogeBetsPrediction.Rounds(previousEpoch);
  const lastRound = await dogeBetsPrediction.Rounds(lastEpoch);
  const RoundLockPrice = parseFloat((parseFloat(ethers.utils.formatEther(previousRound.lockPrice)) * 1e10).toFixed(2));
  const lastRoundLockPrice = parseFloat((parseFloat(ethers.utils.formatEther(lastRound.lockPrice)) * 1e10).toFixed(2));
  const RoundClosePrice = parseFloat((parseFloat(ethers.utils.formatEther(previousRound.closePrice)) * 1e10).toFixed(2));
  const lastRoundClosePrice = parseFloat((parseFloat(ethers.utils.formatEther(lastRound.closePrice)) * 1e10).toFixed(2));
  const priceDifference = parseFloat((RoundClosePrice - RoundLockPrice).toFixed(2));
  const priceDifferenceLast = parseFloat((lastRoundClosePrice - lastRoundLockPrice).toFixed(2));

  return {
    epoch, currentTimeInSeconds, roundStartTimeInSeconds, currentEpochStartLockTime, NextEpochStartLockTime,
    currentEpochCloseEndTime, nextEpochStartLockTimeInSeconds, RoundTime_5m, nextEpochTime, previousEpoch,
    previousRound, RoundLockPrice, RoundClosePrice, priceDifference, lastEpoch, lastRound, priceDifferenceLast,
    lastRoundLockPrice, lastRoundClosePrice
  };
};

export const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export function formatTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  let hours = date.getHours();
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const seconds = date.getSeconds().toString().padStart(2, '0');
  const amPm = hours >= 12 ? 'PM' : 'AM';
  hours = hours % 12 || 12;
  return `${hours.toString().padStart(2, '0')}:${minutes}:${seconds} ${amPm}`;
}

export async function publishEpochToFlask(epoch: number, startTimestampSec: number) {
  try {
    // Node 18+ has global fetch. If you're on Node 16, install node-fetch or use axios.
    const res = await fetch("http://127.0.0.1:5001/update_round", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epoch,
        timestamp: new Date(startTimestampSec * 1000).toISOString(), // UTC ISO
      }),
    });

    // optional: swallow non-200s
    // if (!res.ok) console.log("Flask update_round failed:", res.status);
  } catch (e) {
    // Do not crash bot if Flask is off
    // console.log("publishEpochToFlask error:", e);
  }
}

export const getCurrentRoundDetails = async () => {
  const currentEpoch = await dogeBetsPrediction.currentEpoch();
  const roundDetails = await dogeBetsPrediction.Rounds(currentEpoch);
  const { startTimestamp, lockTimestamp, closeTimestamp, lockPrice, closePrice, closed } = roundDetails;
  return {
    epoch: currentEpoch,
    startTimestamp,
    lockTimestamp,
    closeTimestamp,
    lockPrice,
    closePrice,
    closed: closed ? 'Yes' : 'No'
  };
};

export const placeBet = async (
  dogeBetsPrediction: DogeBetsPrediction,
  wallet: Wallet,
  epoch: number,
  trend: "Bull" | "Bear" | "Neutral",
  bet_amount: number
) => {

  if (trend === 'Bear') {
    try {
      const tx = await dogeBetsPrediction.user_BetBear(epoch, { value: parseEther(bet_amount.toString()) });
      await tx.wait();
      console.log(brightMagenta(`${formatTime(Date.now() / 1000)} Round ${epoch.toString()}:  wager ${bet_amount}   Bear Betting Tx Success.`));
    } catch {
      console.log(magenta(`${formatTime(Date.now() / 1000)} Round ${epoch.toString()}:  wager ${bet_amount}   Bear Betting Tx Error`));
    }
  } else if (trend === 'Bull') {
    try {
      const tx = await dogeBetsPrediction.user_BetBull(epoch, { value: parseEther(bet_amount.toString()) });
      await tx.wait();
      console.log(brightCyan(`${formatTime(Date.now() / 1000)} Round ${epoch.toString()}:  wager ${bet_amount}   Bull Betting Tx Success.`));
    } catch {
      console.log(cyan(`${formatTime(Date.now() / 1000)} Round ${epoch.toString()}:  wager ${bet_amount}   Bull Betting Tx Error`));
    }
  } else if (trend === 'Neutral') {
    console.log(yellow(`${formatTime(Date.now() / 1000)} Round ${epoch.toString()}: Neutral Limits Reached. Skipping round...`));
  } else {
    console.log(yellow(`${formatTime(Date.now() / 1000)} Round ${epoch.toString()}: Technical Analysis not definitive enough. Skipping round...`));
  }
};

interface TrendRecord {
  timestamp: string;
  confidence: number;
  trend: 'Bull' | 'Bear' | 'Neutral';
}

export const executeBetBasedOnTrend = async (epoch: number) => {
  try {
    if (!fs.existsSync(TREND_LOG_FILE)) {
      console.log('⚠️ Trend log file not found.');
      return;
    }
    const content = fs.readFileSync(TREND_LOG_FILE, 'utf-8').trim();
    if (!content) {
      console.log('⚠️ Trend log file is empty.');
      return;
    }
    const records: TrendRecord[] = JSON.parse(content);
    if (!Array.isArray(records) || records.length === 0) {
      console.log('⚠️ No trend records found.');
      return;
    }
    const latest = records[records.length - 1];
    const { trend, confidence } = latest;
    const absSlope = Math.abs(confidence);
    const multiplier = (absSlope * 0.0725).toFixed(6);
    const bet_amount = parseFloat(wager) * parseFloat(multiplier);
    console.log(`**** Wager ${epoch.toString()} Round ****   Trend: ${trend},   Wager ${wager} x Multiplier: ${multiplier} = Bet Amount: ${bet_amount}`);
    await placeBet(dogeBetsPrediction, wallet, epoch, trend, bet_amount);
  } catch (error) {
    console.error('❌ Error reading or processing trend data:', error);
  }
};
