import * as fs from 'fs';
import * as path from 'path';

// Define the path to your JSON file
const TREND_LOG_FILE = path.join(__dirname, 'json/DB_rounds_trend.json');

// Define the shape of a trend record
interface TrendRecord {
  timestamp: string;
  confidence: number;
  trend: 'Bull' | 'Bear' | 'Neutral';
}

// Read and display the most recent entry
function readLatestTrend() {
  try {
    if (!fs.existsSync(TREND_LOG_FILE)) {
      console.log('Trend log file not found.');
      return;
    }

    const content = fs.readFileSync(TREND_LOG_FILE, 'utf-8').trim();
    if (!content) {
      console.log('Trend log file is empty.');
      return;
    }

    const records: TrendRecord[] = JSON.parse(content);

    if (!Array.isArray(records) || records.length === 0) {
      console.log('No trend records found.');
      return;
    }

    const latest = records[records.length - 1];
    console.log(`Timestamp: ${latest.timestamp}`);
    console.log(`Confidence: ${latest.confidence}%`);
    console.log(`Trend: ${latest.trend}`);
  } catch (error) {
    console.error('❌ Error reading or parsing trend data:', error);
  }
}

// Run the function
readLatestTrend();
