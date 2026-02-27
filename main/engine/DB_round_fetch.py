# \main\engine\DB_round_fetch.py
#
# Purpose:
#   Fetch prev/curr/next epoch + timestamps for logging / downstream modules.
#
# Update (per your request):
#   - Adds "Known Results" (ground truth) block using round_record.json.
#   - Prints last 10 known epochs, then prints Epoch Data (prev/curr/next).
#
# Notes:
#   round_record.json records keys like:
#     previousEpoch, startPrice, endPrice, priceDifference, nextEpoch, nextEpochTime
#   (used as ground truth history)
#
# ---------------------------------------------------------------------

import csv
import json
import logging
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# File locations (TREND_LOG_FILE and NEXT_EPOCH_FILE are unchanged)
# ---------------------------------------------------------------------
TREND_LOG_FILE  = Path(r"/main/engine/ts/json/DB_rounds_trend.json")
NEXT_EPOCH_FILE = Path(r"/main/engine/ts/json/pre_round_record.json")
ROUND_RECORD    = Path(r"/main/engine/ts/json/round_record.json")

# New unified processed CSV (replaces both old EPOCH_TIME_FILE + processed path)
PROCESSED_DATA_FILE = Path(
    r"E:\Trading_Bot_V1.0\DogeBets\main\engine\graphing\logs\DB_PROCESSED_DATA.csv"
)

# (Optional alias for compatibility with older code that expects EPOCH_TIME_FILE)
EPOCH_TIME_FILE = PROCESSED_DATA_FILE


# ---------------------------------------------------------------------
# Logging config
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%I:%M:%S %p",
    force=True
)

def _to_json_style(ts: str) -> str:
    """
    Convert ISO timestamp like:
        2026-02-15T12:51:22-05:00
    into JSON-style:
        02/15/2026  12:51:22 PM
    """
    if not ts or not isinstance(ts, str):
        return ts

    try:
        s = ts.strip()
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.strftime("%m/%d/%Y  %I:%M:%S %p")
    except Exception:
        return ts


def _is_int_str(s: str) -> bool:
    return isinstance(s, str) and s.strip().isdigit()


# ---------------------------------------------------------------------
# Round-record helpers (ground truth / known results)
# ---------------------------------------------------------------------
def _resolve_round_record_path() -> Path:
    """
    Allow both your absolute /main/... path and a local path beside the script.

    In your repo, DB_round_fetch.py typically lives in: main/engine/
    So round_record.json is expected at: main/engine/ts/json/round_record.json
    """
    if ROUND_RECORD.exists():
        return ROUND_RECORD

    here = Path(__file__).resolve().parent
    alt = (here / "ts" / "json" / "round_record.json").resolve()
    return alt


def load_known_results(limit: int = 10):
    """Return the last N round records from round_record.json as a list of dicts."""
    path = _resolve_round_record_path()
    if not path.exists():
        logging.warning(f"⚠️ Missing round_record.json: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            logging.warning("⚠️ round_record.json is not a list/dict JSON structure.")
            return []

        # Ensure chronological by previousEpoch if present
        def _key(d):
            try:
                return int(d.get("previousEpoch"))
            except Exception:
                return -1

        data_sorted = sorted(data, key=_key)
        return data_sorted[-max(1, int(limit)) :]

    except Exception as e:
        logging.error(f"💥 Failed to read round_record.json: {e}")
        return []


def log_known_results(limit: int = 10):
    """Log the last N known results in the format you showed."""
    rows = load_known_results(limit=limit)
    logging.info("Known Results:")
    if not rows:
        logging.info("(none)")
        return

    for r in rows:
        ep = r.get("previousEpoch")
        sp = r.get("startPrice")
        epc = r.get("endPrice")
        diff = r.get("priceDifference")
        logging.info(f"epoch {ep}, startPrice {sp}, endPrice {epc}, result {diff}")


# ---------------------------------------------------------------------
# Optional: module-level cache (safe + very effective in tight loops)
# ---------------------------------------------------------------------
_LAST_FETCH_MTIME = None
_LAST_FETCH_RESULT = None


def fetch_last_epoch_info():
    """
    Returns:
        (prev_epoch, prev_epoch_time, curr_epoch, curr_epoch_time, next_epoch, next_epoch_time)

    Source: DB_PROCESSED_DATA.csv (graphrounds logger output)

    Mapping:
      - curr_epoch         <- last LIVE row "epoch" (row_id numeric)
      - next_epoch         <- same row "next_epoch"
      - next_epoch_time    <- same row "next_round_time_est"
      - curr_epoch_time    <- last row where boundary_epoch == curr_epoch, boundary_ts_est present
      - prev_epoch_time    <- last row where boundary_epoch == (curr_epoch-1), boundary_ts_est present
    """
    global _LAST_FETCH_MTIME, _LAST_FETCH_RESULT

    safe = (None, "N/A", None, "N/A", None, "N/A")

    try:
        path = PROCESSED_DATA_FILE
        if not path.exists():
            logging.warning(f"⚠️ Missing processed CSV: {path}")
            return safe

        # Cache: if file hasn't changed, don't reread it every loop.
        mtime = path.stat().st_mtime
        if _LAST_FETCH_MTIME == mtime and _LAST_FETCH_RESULT is not None:
            return _LAST_FETCH_RESULT

        last_live = None
        # boundary_epoch(int) -> latest boundary_ts_est(str)
        last_boundary_ts = {}

        with open(path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for r in reader:
                # Track last LIVE row (row_id numeric + epoch present)
                rid = (r.get("row_id") or "").strip()
                ep = (r.get("epoch") or "").strip()
                if rid.isdigit() and ep != "":
                    last_live = r

                # Track latest boundary time per boundary_epoch
                be = (r.get("boundary_epoch") or "").strip()
                bt = (r.get("boundary_ts_est") or "").strip()
                if bt and _is_int_str(be):
                    last_boundary_ts[int(be)] = bt

        if last_live is None:
            logging.warning("⚠️ No live rows found (row_id numeric + epoch present).")
            return safe

        curr_epoch_s = (last_live.get("epoch") or "").strip()
        next_epoch_s = (last_live.get("next_epoch") or "").strip()
        next_epoch_time = (last_live.get("next_round_time_est") or "").strip() or "N/A"

        if not _is_int_str(curr_epoch_s):
            logging.warning("⚠️ Last live row has non-numeric epoch.")
            return safe

        curr_epoch = int(curr_epoch_s)
        prev_epoch = curr_epoch - 1
        next_epoch = int(next_epoch_s) if _is_int_str(next_epoch_s) else None

        curr_epoch_time = last_boundary_ts.get(curr_epoch, "N/A")
        prev_epoch_time = last_boundary_ts.get(prev_epoch, "N/A")

        result = (
            prev_epoch, prev_epoch_time,
            curr_epoch, curr_epoch_time,
            next_epoch, next_epoch_time
        )

        _LAST_FETCH_MTIME = mtime
        _LAST_FETCH_RESULT = result
        return result

    except Exception as e:
        logging.error(f"💥 Exception in fetch_last_epoch_info(): {e}")
        return safe


if __name__ == "__main__":
    try:
        # 1) Ground-truth history (from round_record.json)
        log_known_results(limit=20)

        logging.info("----------------------------------------------------------------------------")
        logging.info("Epoch Data:")

        prev_epoch, prev_epoch_time, curr_epoch, curr_epoch_time, next_epoch, next_epoch_time = fetch_last_epoch_info()

        prev_epoch_time = _to_json_style(prev_epoch_time)
        curr_epoch_time = _to_json_style(curr_epoch_time)
        next_epoch_time = _to_json_style(next_epoch_time)

        logging.info(f"prev_epoch: {prev_epoch} | prev_epoch_time: {prev_epoch_time}")
        logging.info(f"curr_epoch: {curr_epoch} | curr_epoch_time: {curr_epoch_time}")
        logging.info(f"next_epoch: {next_epoch} | next_epoch_time: {next_epoch_time}")

    except Exception as e:
        logging.error(f"💥 Exception in main execution: {e}")
