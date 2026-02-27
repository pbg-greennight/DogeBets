const MAX_ENTRIES = 100;
let dataEntriesIND: DataEntryIND[] = [];
let dataEntriesTVS: DataEntryTVS[] = [];

setInterval(async () => {
  try {
const tradingViewData = await analyzeindicators();
if (tradingViewData) {
  const newDataEntryTVS: DataEntryTVS = {
    buyBNB1m: tradingViewData.BuyBNB1m,
    sellBNB1m: tradingViewData.SellBNB1m,
    neutralBNB1m: tradingViewData.NeutralBNB1m,
    buyBTC1m: tradingViewData.BuyBTC1m,
    sellBTC1m: tradingViewData.SellBTC1m,
    neutralBTC1m: tradingViewData.NeutralBTC1m,
    historicalData: dataEntriesTVS.slice(0, 100).map(entry => ({
    buyBNB1m: entry.buyBNB1m,
    sellBNB1m: entry.sellBNB1m,
    neutralBNB1m: entry.neutralBNB1m,
    buyBTC1m: entry.buyBTC1m,
    sellBTC1m: entry.sellBTC1m,
    neutralBTC1m: entry.neutralBTC1m
}))
  };
    dataEntriesTVS.unshift(newDataEntryTVS); // Add new data to the start of the array
    if (dataEntriesTVS.length > MAX_ENTRIES) {
      dataEntriesTVS.pop(); // Remove oldest data to maintain array size
    }
    }
  } catch (error) {
    console.error('Error during processing:', error);
  }
}, 15000);
setInterval(async () => {
  try {
 const indicatorsData = await analyzeindicators();
 if (indicatorsData) {
 const newDataEntryIND: DataEntryIND = {
    RSIBNB1m_0: indicatorsData.RSIBNB1m_0, RSIBTC1m_0: indicatorsData.RSIBTC1m_0, ADXBNB1m_0: indicatorsData.ADXBNB1m_0, ADXBTC1m_0: indicatorsData.ADXBTC1m_0,
    BNB_close: indicatorsData.BNB_close, BNB_volume: indicatorsData.BNB_volume, BNB_high: indicatorsData.BNB_high, BNB_low: indicatorsData.BNB_low,
    BTC_close: indicatorsData.BTC_close, BTC_volume: indicatorsData.BTC_volume, BTC_high: indicatorsData.BTC_high, BTC_low: indicatorsData.BTC_low,

    EMABNB1m_0: indicatorsData.EMABNB1m_0, EMABTC1m_0: indicatorsData.EMABTC1m_0,
    BB_upperBNB1m_0: indicatorsData.BB_upperBNB1m_0, BB_upperBTC1m_0: indicatorsData.BB_upperBTC1m_0, BB_middleBNB1m_0: indicatorsData.BB_middleBNB1m_0, BB_middleBTC1m_0: indicatorsData.BB_middleBTC1m_0, BB_lowerBNB1m_0: indicatorsData.BB_lowerBNB1m_0, BB_lowerBTC1m_0: indicatorsData.BB_lowerBTC1m_0,
    KBNB1m_0: indicatorsData.KBNB1m_0, DBNB1m_0: indicatorsData.DBNB1m_0, DBTC1m_0: indicatorsData.DBTC1m_0, KBTC1m_0: indicatorsData.KBTC1m_0,
  };
  dataEntriesIND.unshift(newDataEntryIND);
    if (dataEntriesIND.length > MAX_ENTRIES) {
      dataEntriesIND.pop(); // Remove oldest data to maintain array size
    }
  }
  } catch (error) {
    console.error('Error during processing:', error);
  }
}, 15000);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import { analyzeindicators, DataEntryIND, dataEntriesIND, HistoricalDataEntry, DataEntryTVS, dataEntriesTVS } from './precalculation';


export const calculation = async (): Promise<string> => {
  return new Promise<string>((resolve, reject) => {
    setInterval(async () => {
      try {
        const dataEntriesTVS: DataEntryTVS[] = [];
        const indicatorsData = await analyzeindicators();
        dataEntriesTVS.push({
    buyBNB1m: indicatorsData.BuyBNB1m || null,
    sellBNB1m: indicatorsData.SellBNB1m || null,
    neutralBNB1m: indicatorsData.NeutralBNB1m || null,
    buyBTC1m: indicatorsData.BuyBTC1m || null,
    sellBTC1m: indicatorsData.SellBTC1m || null,
    neutralBTC1m: indicatorsData.NeutralBTC1m || null,
    historicalData: [], // You can initialize this as an empty array or with appropriate data
  });

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                            Calculations
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
type Accumulator = { sum: number; count: number };
const calculateAverage = (
  entries: DataEntryTVS[],
  key: keyof DataEntryTVS
): number => {
  const { sum, count } = entries.slice(0, 300).reduce((acc: Accumulator, entry: DataEntryTVS) => {
    // Using type assertion to reassure TypeScript that entry[key] is a number
    const value = typeof entry[key] === 'number' ? entry[key] as number : 0;
    return { sum: acc.sum + value, count: acc.count + (typeof entry[key] === 'number' ? 1 : 0) };
  }, { sum: 0, count: 0 });

  return count > 0 ? sum / count : 0;
};


const avgBuyBNB1m = calculateAverage(dataEntriesTVS, 'buyBNB1m');
const avgSellBNB1m = calculateAverage(dataEntriesTVS, 'sellBNB1m');
const avgNeutralBNB1m = calculateAverage(dataEntriesTVS, 'neutralBNB1m');
const avgBuyBTC1m = calculateAverage(dataEntriesTVS, 'buyBTC1m');
const avgSellBTC1m = calculateAverage(dataEntriesTVS, 'sellBTC1m');
const avgNeutralBTC1m = calculateAverage(dataEntriesTVS, 'neutralBTC1m');
const avg_BNB_BSN_CALC =
  (avgSellBNB1m < avgBuyBNB1m && avgSellBNB1m / avgBuyBNB1m > avgNeutralBNB1m) ||
  (avgSellBNB1m > avgBuyBNB1m && avgBuyBNB1m / avgSellBNB1m < avgNeutralBNB1m);
const avg_BTC_BSN_CALC =
  (avgSellBTC1m < avgBuyBTC1m && avgSellBTC1m / avgBuyBTC1m > avgNeutralBNB1m) ||
  (avgSellBTC1m > avgBuyBTC1m && avgBuyBTC1m / avgSellBTC1m < avgNeutralBNB1m);
const avg_BNB_BTC_BSN_CALC =  avg_BTC_BSN_CALC && avg_BTC_BSN_CALC;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                           Result
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
let result = '';

        if ((avgNeutralBTC1m > 38 || avgNeutralBNB1m > 38)) {
          result = 'neutral';
        } else if (avg_BNB_BTC_BSN_CALC) {
          result = 'false';
        } else {
          result = 'true';
        }
  return result;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                           Logs
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  console.log("result", result);
//        resolve(result);
  console.log("avgBuyBNB1m", avgBuyBNB1m, "avgSellBNB1m", avgSellBNB1m, "avgNeutralBNB1m", avgNeutralBNB1m, "avgBuyBTC1m", avgBuyBTC1m, "avgSellBTC1m", avgSellBTC1m, "avgNeutralBTC1m", avgNeutralBTC1m, "avg_BNB_BSN_CALC", avg_BNB_BSN_CALC, "avg_BTC_BSN_CALC", avg_BTC_BSN_CALC, "avg_BNB_BTC_BSN_CALC", avg_BNB_BTC_BSN_CALC);

//  console.log("====================================================================================================================================================================================");




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      } catch (error) {
        console.error('Error during processing:', error);
        reject(error);
      }
    }, 15000);
  });
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import { red, green, blue, yellow, magenta, cyan, white,
         gray, black, bgRed, bgGreen, bgBlue, bgYellow,
         bgMagenta, bgCyan, bgWhite, bgGray, bgBlack } from 'chalk'
import { analyzeindicators } from './indicators';
import { analyzeTradingView } from './tradingviewscan';
import moment from 'moment-timezone';

function formatEasternTime() {
  const estTime = moment().tz("America/New_York");
  return estTime.format("MM/DD/YY hh:mm:ss A");
}
const currentTime = formatEasternTime();


async function indicators() {
  try {
    const indicatorsData = await analyzeindicators();
    const tradingViewData = await analyzeTradingView();
  } catch (error) {
    console.error(red('Error Retrieving indicator Data:', error));
 }
}
indicators();


export const analyzecalculation = async (): Promise<'true' | 'false' | 'neutral'> => {

export interface DataEntry {
    RSIBNB1m_0: number | null; RSIBTC1m_0: number | null;
    BNB_close: number[] | null; BNB_volume: number[] | null; BNB_high: number[] | null; BNB_low: number[] | null;
    BTC_close: number[] | null; BTC_volume: number[] | null; BTC_high: number[] | null; BTC_low: number[] | null;
    ADXBNB1m_0: { adx: number | null; pdi: number | null; mdi: number | null; } | null;
    ADXBTC1m_0: { adx: number | null; pdi: number | null; mdi: number | null; } | null;
    EMABNB1m_0: number | null; EMABTC1m_0: number | null;
    BB_upperBNB1m_0: number | null; BB_upperBTC1m_0: number | null; BB_middleBNB1m_0: number | null; BB_middleBTC1m_0: number | null; BB_lowerBNB1m_0: number | null; BB_lowerBTC1m_0: number | null;
    KBNB1m_0: number | null; DBNB1m_0: number | null; DBTC1m_0: number | null; KBTC1m_0: number | null;
    BuyBNB1m: number | null; SellBNB1m: number | null; NeutralBNB1m: number | null;
    BuyBTC1m: number | null; SellBTC1m: number | null; NeutralBTC1m: number | null;

}

const MAX_ENTRIES = 20;
let dataEntries: DataEntry[] = [];
let entry20: DataEntry | undefined; let entry19: DataEntry | undefined; let entry18: DataEntry | undefined; let entry17: DataEntry | undefined; let entry16: DataEntry | undefined;
let entry15: DataEntry | undefined; let entry14: DataEntry | undefined; let entry13: DataEntry | undefined; let entry12: DataEntry | undefined; let entry11: DataEntry | undefined;
let entry10: DataEntry | undefined; let entry9: DataEntry | undefined; let entry8: DataEntry | undefined; let entry7: DataEntry | undefined; let entry6: DataEntry | undefined;
let entry5: DataEntry | undefined; let entry4: DataEntry | undefined; let entry3: DataEntry | undefined; let entry2: DataEntry | undefined; let entry1: DataEntry | undefined;
let indicatorsData: any;

setInterval(async () => {
    try {
        const indicatorsData = await analyzeindicators();
        const tradingViewData = await analyzeTradingView();
    console.log('indicatorsData:', indicatorsData);
    console.log('tradingViewData:', tradingViewData);
        const data: DataEntry = { ...indicatorsData, ...tradingViewData };
//        console.log("Time", currentTime, "New Data Entry:", data); // Add this line to debug
          dataEntries.push(data);

        if (dataEntries.length > MAX_ENTRIES) {
            dataEntries.shift();
        }

        if (dataEntries.length >= MAX_ENTRIES) {
            const entries = dataEntries.slice(-6);
            entry20 = dataEntries[dataEntries.length - 1]; entry19 = dataEntries[dataEntries.length - 2]; entry18 = dataEntries[dataEntries.length - 3];
            entry17 = dataEntries[dataEntries.length - 4]; entry16 = dataEntries[dataEntries.length - 5]; entry15 = dataEntries[dataEntries.length - 6];
            entry14 = dataEntries[dataEntries.length - 7]; entry13 = dataEntries[dataEntries.length - 8]; entry12 = dataEntries[dataEntries.length - 9];
            entry11 = dataEntries[dataEntries.length - 10]; entry10 = dataEntries[dataEntries.length - 11]; entry9 = dataEntries[dataEntries.length - 12];
            entry8 = dataEntries[dataEntries.length - 13]; entry7 = dataEntries[dataEntries.length - 14]; entry6 = dataEntries[dataEntries.length - 15];
            entry5 = dataEntries[dataEntries.length - 16]; entry4 = dataEntries[dataEntries.length - 17]; entry3 = dataEntries[dataEntries.length - 18];
            entry2 = dataEntries[dataEntries.length - 19]; entry1 = dataEntries[dataEntries.length - 20];

type Accumulator = { sum: number; count: number };
   const calculateAverage = ( entries: DataEntry[], key: keyof DataEntry ): number => {
   const { sum, count } = entries.slice(0, 20).reduce((acc: Accumulator, entry: DataEntry) => {
   const value = typeof entry[key] === 'number' ? entry[key] as number : 0;
       return { sum: acc.sum + value, count: acc.count + (typeof entry[key] === 'number' ? 1 : 0) }; }, { sum: 0, count: 0 });
       return count > 0 ? sum / count : 0; };

 const avgEMABNB1m_0 = calculateAverage(dataEntries, 'EMABNB1m_0').toFixed(2); const avgEMABTC1m_0 = calculateAverage(dataEntries, 'EMABTC1m_0').toFixed(2);
 const avgBB_upperBNB1m_0 = calculateAverage(dataEntries, 'BB_upperBNB1m_0').toFixed(2); const avgBB_upperBTC1m_0 = calculateAverage(dataEntries, 'BB_upperBTC1m_0').toFixed(2);
 const avgBB_middleBNB1m_0 = calculateAverage(dataEntries, 'BB_middleBNB1m_0').toFixed(2); const avgBB_middleBTC1m_0 = calculateAverage(dataEntries, 'BB_middleBTC1m_0').toFixed(2);
 const avgBB_lowerBNB1m_0 = calculateAverage(dataEntries, 'BB_lowerBNB1m_0').toFixed(2); const avgBB_lowerBTC1m_0 = calculateAverage(dataEntries, 'BB_lowerBTC1m_0').toFixed(2);
 const avgBuyBNB1m = calculateAverage(dataEntries, 'BuyBNB1m'); const avgSellBNB1m = calculateAverage(dataEntries, 'SellBNB1m'); const avgNeutralBNB1m = calculateAverage(dataEntries, 'NeutralBNB1m');
 const avgBuyBTC1m = calculateAverage(dataEntries, 'BuyBTC1m'); const avgSellBTC1m = calculateAverage(dataEntries, 'SellBTC1m'); const avgNeutralBTC1m = calculateAverage(dataEntries, 'NeutralBTC1m');
 const avg_BNB_BSN_CALC = (avgSellBNB1m < avgBuyBNB1m && avgSellBNB1m / avgBuyBNB1m > avgNeutralBNB1m) || (avgSellBNB1m > avgBuyBNB1m && avgBuyBNB1m / avgSellBNB1m < avgNeutralBNB1m);
 const avg_BTC_BSN_CALC = (avgSellBTC1m < avgBuyBTC1m && avgSellBTC1m / avgBuyBTC1m > avgNeutralBNB1m) || (avgSellBTC1m > avgBuyBTC1m && avgBuyBTC1m / avgSellBTC1m < avgNeutralBNB1m);
 const avg_BNB_BTC_BSN_CALC = avg_BTC_BSN_CALC && avg_BTC_BSN_CALC;
 const avg_B_CALC = (avgSellBNB1m < avgBuyBNB1m && avgSellBNB1m / avgBuyBNB1m > avgNeutralBNB1m) || (avgSellBNB1m > avgBuyBNB1m && avgBuyBNB1m / avgSellBNB1m < avgNeutralBNB1m) &&
                    (avgSellBTC1m < avgBuyBTC1m && avgSellBTC1m / avgBuyBTC1m > avgNeutralBTC1m) || (avgSellBTC1m > avgBuyBTC1m && avgBuyBTC1m / avgSellBTC1m < avgNeutralBTC1m)


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   //export const precalc = calculatePrecalc(exportedData);
   //export const Bet = calculateBet(indicatorsData, exportedData);

 let precalc: 'true' | 'false' = 'false';

    if (
        (5 <= 10)
//        (diffToHigh <= diffToLow)
        ) {
            precalc = 'false';
    } else if (
        (5 <= 10)
//          (diffToHigh >= diffToLow)
    ) {
      precalc = 'true';
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
let Bet: 'true' | 'false' | 'neutral' = 'neutral';

    if (
        (5 <= 6)
//      (diffToHigh >= diffToLow)
    ) {
            Bet = 'neutral';
    } else if (
        (5 <= 10)
//      (diffToHigh <= diffToLow)
        ) {
            Bet = 'false';
    } else if (
        (5 <= 10)
//       (diffToHigh >= diffToLow)
       ) {
            Bet = 'true';
        }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//    console.log(`${formatEasternTime()} BNB_BSN_C ${avg_BNB_BSN_CALC}, BTC_BSN_C ${avg_BTC_BSN_CALC}, BNB_BTC_BSN_C ${avg_BNB_BTC_BSN_CALC}, precalc: ${precalc}, Bet: ${Bet}`);
//    console.log(`${formatEasternTime()} AVG 1m: BuyBNB ${avgBuyBNB1m}, SellBNB ${avgSellBNB1m}, NeutralBNB ${avgNeutralBNB1m}, BuyBTC ${avgBuyBTC1m}, SellBTC ${avgSellBTC1m}, NeutralBTC ${avgNeutralBTC1m}`);
//    console.log(`----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------`);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 }
  } catch (error) {
    console.error('Error during processing:', error);
  }
}, 30000); // 30 seconds

  // Return a Promise that never resolves, indicating that this function runs indefinitely
  return new Promise(() => {});
};