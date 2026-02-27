import fs from 'fs';
import path from 'path';
import moment from 'moment-timezone';

const ROUND_RECORD_PATH = path.join(__dirname, 'json/round_record.json');
const PRE_ROUND_RECORD_PATH = path.join(__dirname, 'json/pre_round_record.json');

function normalizeTimestamp(ts: string): string {
  if (/\d{1,2}\/\d{1,2}\/\d{4}/.test(ts)) return ts;
  const today = moment().tz("America/New_York").format("MM/DD/YYYY");
  const dt = moment.tz(`${today} ${ts}`, "MM/DD/YYYY hh:mm:ss A", "America/New_York");
  if (!dt.isValid()) return ts;

  return dt.format("MM/DD/YYYY  hh:mm:ss A");
}

type RoundRecord = {
  current_timestamp: string;
  previousEpoch: number;
  startPrice: string;
  endPrice: string;
  priceDifference: string;
  nextEpoch: number;
  nextEpochTime: string;
};

type PreRoundRecord = {
  current_timestamp: string;
  nextEpochTime: string;
  nextEpoch: number;
};

// --- Safe JSON array I/O (salvage + atomic write) ---
function readJsonArraySafe<T = any>(file: string): T[] {
  try {
    if (!fs.existsSync(file)) return [];
    const raw = fs.readFileSync(file, 'utf8').trim();
    if (!raw) return [];
    // happy path
    if (raw.startsWith('[')) return JSON.parse(raw);
    // salvage: try to extract the array portion if the file was corrupted
    const i = raw.indexOf('[');
    const j = raw.lastIndexOf(']');
    if (i >= 0 && j > i) return JSON.parse(raw.slice(i, j + 1));
    return [];
  } catch {
    return [];
  }
}

function writeJsonArrayAtomic(file: string, data: any[]) {
  const tmp = file + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2) + '\n', 'utf8');
  fs.renameSync(tmp, file);
}

function dedupeByPreviousEpoch(list: RoundRecord[]): RoundRecord[] {
  const map = new Map<number, RoundRecord>();
  for (const rec of list) map.set(rec.previousEpoch, rec); // last one wins
  return Array.from(map.values()).sort((a, b) => a.previousEpoch - b.previousEpoch);
}

export function saveRoundData(
  currentTime: string,
  previousEpoch: number,
  roundLockPrice: number,
  roundClosePrice: number,
  priceDifference: number,
  nextEpoch: number,
  nextEpochTime: string
) {
  const roundRecord: RoundRecord = {
    current_timestamp: normalizeTimestamp(currentTime),
    previousEpoch: Number(previousEpoch),
    startPrice: `$${Number(roundLockPrice).toFixed(2)}`,
    endPrice: `$${Number(roundClosePrice).toFixed(2)}`,
    priceDifference: `$${Number(priceDifference).toFixed(2)}`,
    nextEpoch: Number(nextEpoch),
    nextEpochTime: normalizeTimestamp(nextEpochTime),
  };

  // read → mutate → de-dupe → truncate → atomic write
  let existing = readJsonArraySafe<RoundRecord>(ROUND_RECORD_PATH);
  existing.push(roundRecord);
 // existing = dedupeByPreviousEpoch(existing);

  if (existing.length > 150000) existing = existing.slice(-150000);

  writeJsonArrayAtomic(ROUND_RECORD_PATH, existing);
}

export function PreRoundData(
  currentTime: string,
  nextEpochTime: string,
  nextEpoch: number
) {
  const preRoundRecord: PreRoundRecord = {
    current_timestamp: normalizeTimestamp(currentTime),
    nextEpochTime: normalizeTimestamp(nextEpochTime),
    nextEpoch: Number(nextEpoch)
  };

  let existing = readJsonArraySafe<PreRoundRecord>(PRE_ROUND_RECORD_PATH);
  existing.push(preRoundRecord);

  // Keep it lean (e.g., last 150k)
  if (existing.length > 150000) existing = existing.slice(-150000);

  writeJsonArrayAtomic(PRE_ROUND_RECORD_PATH, existing);
}

