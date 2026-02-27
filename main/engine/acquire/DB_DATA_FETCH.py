"""
DB_DATA_FETCH.py

Fetches 1m BTCUSDT OHLCV data from Binance every ~2.5 seconds and appends it
to DB_OCHLV_DATA.csv (or DB_OCHLV_DATA_test.csv if you keep this path).

Key behavior:

- Timestamp in CSV is the *local wall-clock time* (America/Toronto) when the
  fetch occurs, including full date + seconds, e.g.:
      "2025-11-14 11:57:03 AM"
- This matches the style of your screen log (HH:MM:SS AM/PM) and gives DB_data_epoch_fetch
  something with both date and seconds to align against DB_epoch_time.csv.
"""

import os
import time
import logging
from datetime import datetime

import ccxt
import pytz
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Symbol & timeframe
SYMBOL = "BTCUSDT"          # Binance raw symbol (ccxt will map as needed)
TIMEFRAME = "1m"

# Fetch interval in seconds (approx 2.5s)
FETCH_INTERVAL_SEC = 2.5

# Timezone (America/Toronto as per your environment)
EST = pytz.timezone("America/Toronto")

# CSV file to store OCHLV data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Change this to "DB_OCHLV_DATA.csv" when you want to use the real file
OCHLV_CSV_PATH = os.path.join(BASE_DIR, "csv", "DB_OCHLV_DATA.csv")

# CSV columns (must match what DB_data_epoch_fetch.py expects)
CSV_COLUMNS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]

# ---------------------------------------------------------------------------
# Logging setup (keep same style: HH:MM:SS AM/PM - LEVEL - message)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%I:%M:%S %p",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_ochlv_csv_exists() -> None:
    """Ensure DB_OCHLV_DATA.csv exists AND has a correct header.

    Important: if the file already exists but was created without a header (your current case),
    pandas will treat the first data row as column names and your trend engine will break.
    """
    # Create if missing
    if not os.path.exists(OCHLV_CSV_PATH):
        df = pd.DataFrame(columns=CSV_COLUMNS)
        df.to_csv(OCHLV_CSV_PATH, index=False)
        logger.info(f"Created new OCHLV CSV file at: {OCHLV_CSV_PATH}")
        return

    # If exists: verify header
    try:
        with open(OCHLV_CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
            first = (f.readline() or "").strip()
        # If first line doesn't look like the expected header, prepend header.
        if not first.lower().startswith("timestamp,"):
            logger.warning(
                "OCHLV CSV exists but appears to be missing a header. Prepending header now. first_line=%r",
                first[:120],
            )
            # Read entire file and rewrite with header + existing contents.
            # (This is safe because we only do it once when the issue is detected.)
            with open(OCHLV_CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
                body = f.read()
            with open(OCHLV_CSV_PATH, "w", encoding="utf-8", errors="ignore") as f:
                f.write(",".join(CSV_COLUMNS) + "\n")
                f.write(body)
    except Exception as e:
        logger.error(f"Failed to validate/prepend header for OCHLV CSV: {e}")


def init_exchange() -> ccxt.binance:
    """
    Initialize the Binance exchange via ccxt.
    """
    exchange = ccxt.binance({
        "enableRateLimit": True,
        # If you use testnet or custom options, plug them here.
    })
    return exchange


def fetch_latest_ohlcv(exchange: ccxt.binance):
    """
    Fetch the latest 1m OHLCV candle for SYMBOL from Binance.

    Returns:
        (timestamp_ms, open, high, low, close, volume) or None on error.

    Note: we still return the candle timestamp in case you later want to log it,
    but for the CSV 'Timestamp' we will use the local wall-clock time.
    """
    try:
        candles = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=1)
        if not candles:
            return None

        ts_ms, o, h, l, c, v = candles[0]
        return ts_ms, float(o), float(h), float(l), float(c), float(v)
    except Exception as e:
        logger.error(f"Error fetching OHLCV data: {e}")
        return None


def append_ochlv_row(local_timestamp_str: str,
                     o: float, h: float, l: float, c: float, v: float) -> None:
    """
    Append a single OHLCV row to DB_OCHLV_DATA.csv.

    local_timestamp_str is your local (America/Toronto) timestamp string,
    with *date + time + seconds*:
        'YYYY-MM-DD HH:MM:SS AM/PM'
    """
    now_est = datetime.now(EST)
    local_timestamp_str = now_est.strftime("%Y-%m-%d %I:%M:%S %p")

    row = {
        "Timestamp": local_timestamp_str,
        "Open": o,
        "High": h,
        "Low": l,
        "Close": c,
        "Volume": v,
    }

    df_row = pd.DataFrame([row], columns=CSV_COLUMNS)
    df_row.to_csv(OCHLV_CSV_PATH, mode="a", header=False, index=False)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Main fetch loop:
    - Ensure CSV exists
    - Initialize exchange
    - Every ~2.5s:
        - Fetch latest OHLCV
        - Build local EST timestamp from *current wall-clock* (not candle ts)
        - Log Open/High/Low/Close/Volume
        - Append to DB_OCHLV_DATA.csv

    Result:
    - The log timestamps (from logging) and the CSV 'Timestamp' both refer
      to the same local time, with second precision and full date.
    """
    ensure_ochlv_csv_exists()
    exchange = init_exchange()

    logger.info(f"Starting OCHLV fetch loop. Writing to: {OCHLV_CSV_PATH}")
    logger.info(f"Symbol: {SYMBOL}, timeframe: {TIMEFRAME}, interval: {FETCH_INTERVAL_SEC}s")

    while True:
        loop_start = time.time()

        data = fetch_latest_ohlcv(exchange)
        if data is not None:
            ts_ms, o, h, l, c, v = data

            # --- LOCAL TIMESTAMP FROM WALL-CLOCK ---
            # Use the actual fetch time in America/Toronto for the CSV.
            now_est = datetime.now(EST)
            local_timestamp_str = now_est.strftime("%Y-%m-%d %I:%M:%S %p")

            # Log O/H/L/C/V (logger's own timestamp will match now_est)
            logger.info(
                "Open:%.2f High:%.2f Low:%.2f Close:%.2f Volume:%.2f",
                o, h, l, c, v
            )

            # Append to CSV with full date + time + seconds
            append_ochlv_row(local_timestamp_str, o, h, l, c, v)

        # Sleep the remainder of the interval
        elapsed = time.time() - loop_start
        sleep_for = max(0.0, FETCH_INTERVAL_SEC - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    run()
