# DogeBets/main/engine/acquire/DB_epoch_timing.py

import json
from pathlib import Path
import os
import time
import logging
import csv
from datetime import datetime
from typing import List, Dict, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - INFO - %(message)s",
    datefmt="%I:%M:%S %p",
    force=True,
)

# ----- Paths (resolved absolutely to avoid CWD issues) -----
SCRIPT_DIR = Path(__file__).resolve().parent
ROUND_RECORD_FILE = (SCRIPT_DIR / "../ts/json/round_record.json").resolve()
POST_ROUND_RECORD_FILE = (SCRIPT_DIR / "../ts/json/pre_round_record.json").resolve()
EPOCH_CSV_FILE = (SCRIPT_DIR / "../acquire/csv/DB_epoch_time.csv").resolve()
EPOCH_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

# ----- Polling -----
CHECK_INTERVAL = 2.5  # seconds between checks

# ----- Fields & behavior -----
# We attach the round summary (previousEpoch/startPrice/endPrice/priceDifference)
# to the row whose **Current Epoch == round_data["nextEpoch"]**.
NEW_FIELDS = ["previousEpoch", "startPrice", "endPrice", "priceDifference"]
BASE_FIELDS = ["Timestamp", "Current Epoch", "Next Epoch", "Next Epoch Time"]
ALL_FIELDS  = BASE_FIELDS + NEW_FIELDS


# =====================================================================
# CSV utilities
# =====================================================================

def _ensure_or_upgrade_header(filename: Path, fieldnames: List[str]) -> None:
    """Create CSV with fieldnames if missing; otherwise add missing columns in place."""
    filename = Path(filename)
    if not filename.exists():
        with filename.open(mode="w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        logging.info("🧾 Created epoch CSV with header at: %s", filename)
        return

    with filename.open(mode="r", newline="") as f:
        reader = csv.DictReader(f)
        existing = reader.fieldnames or []
        rows = list(reader)

    missing = [c for c in fieldnames if c not in existing]
    if not missing:
        return

    with filename.open(mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=existing + missing)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in existing}
            for m in missing:
                out[m] = ""
            writer.writerow(out)
    logging.info("🧾 Upgraded epoch CSV header with new fields %s at: %s", missing, filename)


def _read_csv(filename: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with Path(filename).open(mode="r", newline="") as f:
        reader = csv.DictReader(f)
        return (reader.fieldnames or []), list(reader)


def _write_csv(filename: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with Path(filename).open(mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(out)


def _dedupe_by_current_epoch(filename: Path) -> None:
    """Keep only the latest row per 'Current Epoch'. Latest = last seen in file order."""
    filename = Path(filename)
    if not filename.exists():
        return
    fieldnames, rows = _read_csv(filename)
    if not rows:
        return

    by_curr: Dict[str, Dict[str, str]] = {}
    for r in rows:
        key = str(r.get("Current Epoch", "")).strip()
        if not key:
            continue
        by_curr[key] = r  # overwrite to keep latest

    def _ts_val(r: Dict[str, str]) -> float:
        ts = r.get("Timestamp", "")
        try:
            return datetime.strptime(ts, "%Y-%m-%d %I:%M:%S %p").timestamp()
        except Exception:
            return 0.0

    deduped = sorted(by_curr.values(), key=_ts_val)

    # Make sure we write with ALL_FIELDS columns
    target_fields = list(ALL_FIELDS)
    # Also include any unknown extras that might exist already
    for k in fieldnames:
        if k not in target_fields:
            target_fields.append(k)

    _write_csv(filename, target_fields, deduped)
    logging.info("🧹 Deduped CSV by Current Epoch → kept %d unique rows", len(deduped))


# =====================================================================
# Helpers for epoch math
# =====================================================================

def _to_int_or_none(v: Optional[object]) -> Optional[int]:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        return int(float(s))  # handles strings like "264731" or numbers
    except Exception:
        return None


def _resolve_epochs(round_data: Optional[dict], pre_round_data: Optional[dict]) -> Tuple[str, str]:
    """Return (current_epoch, next_epoch) as strings.
    current_epoch := round_data.nextEpoch (required to anchor rows)
    next_epoch    := force to (current_epoch + 1). If pre_round_data.nextEpoch matches, we keep it; otherwise we correct it.
    """
    curr_i = _to_int_or_none(round_data.get("nextEpoch") if round_data else None)
    pre_nxt_i  = _to_int_or_none(pre_round_data.get("nextEpoch") if pre_round_data else None)

    if curr_i is None:
        # Fallback: if only pre-round exists, use it as current and +1 as next
        curr_i = pre_nxt_i
    # Compute expected next
    nxt_i = (curr_i + 1) if curr_i is not None else None

    # If pre-round gave a matching value, it's fine; otherwise enforce curr+1
    if pre_nxt_i is not None and curr_i is not None and pre_nxt_i == curr_i + 1:
        nxt_i = pre_nxt_i

    curr_s = str(curr_i) if curr_i is not None else "N/A"
    nxt_s  = str(nxt_i)  if nxt_i  is not None else "N/A"
    return curr_s, nxt_s


# =====================================================================
# Backfill + UPSERTs
# =====================================================================

def _upsert_epoch_row(current_epoch: str, next_epoch: str, next_epoch_time: str, filename: Path) -> None:
    """Upsert a single row keyed by 'Current Epoch'. Updates Next fields & Timestamp if exists; else appends."""
    filename = Path(filename)
    _ensure_or_upgrade_header(filename, ALL_FIELDS)

    fieldnames, rows = _read_csv(filename)
    found_idx = -1
    curr_key = str(current_epoch)

    for i, r in enumerate(rows):
        if str(r.get("Current Epoch", "")) == curr_key:
            found_idx = i
            break

    now_ts = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    if found_idx >= 0:
        # Update in place
        rows[found_idx]["Timestamp"] = now_ts
        rows[found_idx]["Current Epoch"] = curr_key
        rows[found_idx]["Next Epoch"] = str(next_epoch)
        rows[found_idx]["Next Epoch Time"] = str(next_epoch_time)
        _write_csv(filename, fieldnames if fieldnames else ALL_FIELDS, rows)
        logging.info("🔁 UPSERT updated row for Current Epoch=%s (Next=%s, NextTime=%s)", curr_key, next_epoch, next_epoch_time)
    else:
        # Append new
        rows.append({
            "Timestamp": now_ts,
            "Current Epoch": curr_key,
            "Next Epoch": str(next_epoch),
            "Next Epoch Time": str(next_epoch_time),
            "previousEpoch": "",
            "startPrice": "",
            "endPrice": "",
            "priceDifference": "",
        })
        _write_csv(filename, fieldnames if fieldnames else ALL_FIELDS, rows)
        logging.info("➕ UPSERT appended row for Current Epoch=%s (Next=%s, NextTime=%s)", curr_key, next_epoch, next_epoch_time)


def _backfill_round_summary(round_data: dict, filename: Path) -> None:
    """ID-based backfill: attach summary to the row whose Current Epoch == round_data['previousEpoch'].
    This matches your corrected CSV where the summary belongs to the just-finished epoch row.
    """
    if not round_data:
        return

    filename = Path(filename)
    _ensure_or_upgrade_header(filename, ALL_FIELDS)

    target_curr = str(round_data.get("previousEpoch", "")).strip()
    if not target_curr:
        logging.info("ℹ️ Backfill skipped: round_data.previousEpoch missing")
        return

    prev_epoch = str(round_data.get("previousEpoch", ""))
    start_price = str(round_data.get("startPrice", ""))
    end_price = str(round_data.get("endPrice", ""))
    price_diff = str(round_data.get("priceDifference", ""))

    fieldnames, rows = _read_csv(filename)
    found_idx = -1
    for i, r in enumerate(rows):
        if str(r.get("Current Epoch", "")) == target_curr:
            found_idx = i
            break

    if found_idx < 0:
        logging.info("ℹ️ Backfill deferred: no row for Current Epoch=%s yet (will fill when that row is created)", target_curr)
        return

    rows[found_idx]["previousEpoch"] = prev_epoch
    rows[found_idx]["startPrice"] = start_price
    rows[found_idx]["endPrice"] = end_price
    rows[found_idx]["priceDifference"] = price_diff

    _write_csv(filename, fieldnames if fieldnames else ALL_FIELDS, rows)
    logging.info(
        "🧩 Backfilled row for Current Epoch=%s → previousEpoch=%s, startPrice=%s, endPrice=%s, priceDifference=%s",
        target_curr, prev_epoch, start_price, end_price, price_diff,
    )
    logging.info("=" * 97)


# =====================================================================
# JSON loading helpers
# =====================================================================
last_round_data = None
last_post_round_data = None
last_round_time_logged = None  # Last time round record was detected


def fetch_json_data(file_path: Path):
    """Fetch and return the latest JSON entry from the given file; supports list or dict payloads."""
    if not file_path.exists():
        logging.error("❌ JSON file %s not found!", file_path)
        return None
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list) and data:
            data = data[-1]  # assume last element is most recent
        if isinstance(data, dict):
            return data
        logging.error("❌ Invalid JSON format in %s. Expected dict or list-of-dicts.", file_path)
        return None
    except json.JSONDecodeError:
        logging.error("❌ Error decoding JSON in %s. It may be corrupt or empty.", file_path)
        return None
    except Exception as e:
        logging.error("❌ Error reading JSON file %s: %s", file_path, e)
        return None


# =====================================================================
# Logging + CSV writing pipeline
# =====================================================================

def format_and_print_data(round_data, pre_round_data, last_round_data_ref, last_post_round_data_ref):
    """
    Formats/logs the latest data. Writes/updates the timing row and backfills round summary by ID.
    This version includes a fix to ensure nextEpochTime always contains a full date.
    """
    global last_round_time_logged

    # Detect new round record
    new_round_record_detected = bool(round_data and round_data != last_round_data_ref)
    new_pre_round_record_detected = bool(pre_round_data and pre_round_data != last_post_round_data_ref)

    # -------------------------------------------------------------------------
    # 1) NEW ROUND RECORD SECTION
    # -------------------------------------------------------------------------
    if new_round_record_detected:
        last_round_time_logged = datetime.now()

        logging.info("🟢 New Round Record Data Entry Detected!")
        logging.info("🔹 New Round Record Data:")
        logging.info(
            "Round Time: %s, Previous Epoch: %s, Start Price: %s, End Price: %s, Price Difference: %s, Next Epoch: %s",
            last_round_time_logged.strftime("%Y-%m-%d %I:%M:%S %p"),
            round_data.get("previousEpoch", "N/A"),
            round_data.get("startPrice", "N/A"),
            round_data.get("endPrice", "N/A"),
            round_data.get("priceDifference", "N/A"),
            round_data.get("nextEpoch", "N/A"),
        )
        logging.info("-" * 97)

    # -------------------------------------------------------------------------
    # 2) NEW PRE-ROUND RECORD SECTION
    # -------------------------------------------------------------------------
    if new_pre_round_record_detected:
        pre_round_time_logged = datetime.now()

        logging.info("🟢 New Pre Round Record Data Entry Detected!")
        logging.info("🔹 Pre Round Record Data:")

        if last_round_time_logged:
            try:
                # Ensure the timing of the two events is reasonable
                time_diff = abs((pre_round_time_logged - last_round_time_logged).total_seconds())

                if time_diff <= 30:
                    # ---------------------------------------------------------
                    # Epoch Resolution
                    # ---------------------------------------------------------
                    current_epoch, next_epoch = _resolve_epochs(round_data, pre_round_data)

                    # ---------------------------------------------------------
                    # FIX: Normalize nextEpochTime to ensure FULL DATE exists
                    # ---------------------------------------------------------
                    raw_next_time = str(pre_round_data.get("nextEpochTime", "")).strip()

                    # Determine if a date already exists in the timestamp
                    # Examples detected as having a date:
                    #   "11/16/2025 10:03:39 PM"
                    #   "2025-11-16 22:03:39"
                    has_date = (
                        "-" in raw_next_time[:10] or                    # yyyy-mm-dd
                        ("/" in raw_next_time[:10] and raw_next_time.count("/") >= 2)  # mm/dd/yyyy
                    )

                    if has_date:
                        fixed_next_epoch_time = raw_next_time
                    else:
                        # Timestamp is time-only (e.g. "10:03:39 PM")
                        # Attach today's date from the moment this data was recorded
                        formatted_date = pre_round_time_logged.strftime("%m/%d/%Y")
                        fixed_next_epoch_time = f"{formatted_date}  {raw_next_time}"

                    next_epoch_time = fixed_next_epoch_time
                    # ---------------------------------------------------------

                    # Log the corrected pre-round entry
                    logging.info(
                        "Pre-Round Current Time: %s, Current Epoch: %s, Next Epoch: %s, Next Epoch Time: %s",
                        pre_round_time_logged.strftime("%Y-%m-%d %I:%M:%S %p"),
                        current_epoch,
                        next_epoch,
                        next_epoch_time,
                    )
                    logging.info("-" * 97)

                    # ---------------------------------------------------------
                    # UPSERT TIMING ROW FOR THIS CURRENT EPOCH
                    # ---------------------------------------------------------
                    _upsert_epoch_row(
                        str(current_epoch),
                        str(next_epoch),
                        str(next_epoch_time),
                        filename=EPOCH_CSV_FILE
                    )

                    # ---------------------------------------------------------
                    # BACKFILL ROUND SUMMARY ROW FOR PREVIOUS EPOCH DATA
                    # ---------------------------------------------------------
                    _backfill_round_summary(
                        round_data,
                        filename=EPOCH_CSV_FILE
                    )

            except Exception as e:
                logging.error("❌ Error calculating time difference: %s", e)


# =====================================================================
# JSON monitor loop
# =====================================================================

def monitor_json_files():
    global last_round_data, last_post_round_data

    # One-time cleanup on startup to collapse any existing duplicates
    _ensure_or_upgrade_header(EPOCH_CSV_FILE, ALL_FIELDS)
    _dedupe_by_current_epoch(EPOCH_CSV_FILE)

    while True:
        round_data = fetch_json_data(ROUND_RECORD_FILE)
        post_round_data = fetch_json_data(POST_ROUND_RECORD_FILE)

        # Detect a new round record and wait briefly for its paired pre-round
        new_round_record_detected = bool(round_data and round_data != last_round_data)

        if new_round_record_detected:
            # Wait up to ~40s for pre-round data to arrive
            for attempt in range(4):  # 4 * 10s = 40s max
                post_round_data = fetch_json_data(POST_ROUND_RECORD_FILE)
                if post_round_data and post_round_data != last_post_round_data:
                    format_and_print_data(round_data, post_round_data, last_round_data, last_post_round_data)
                    last_round_data = round_data
                    last_post_round_data = post_round_data
                    break
                else:
                    logging.error("❌ Pre Round Record Data: data not found. Retrying in 5 seconds.")
                    time.sleep(5)
            else:
                # After timeout, log round anyway, mark pre-round as missing
                format_and_print_data(round_data, None, last_round_data, last_post_round_data)
                last_round_data = round_data  # Prevent infinite loop

        else:
            # No new round; check for a stray new pre-round
            if post_round_data and post_round_data != last_post_round_data:
                format_and_print_data(round_data, post_round_data, last_round_data, last_post_round_data)
                last_post_round_data = post_round_data

        time.sleep(CHECK_INTERVAL)


# =====================================================================
# Epoch CSV Parser (unchanged consumer; ignores extra columns)
# =====================================================================

def parse_csv(file_path):
    next_epoch = []
    next_epoch_times = []
    timestamps = []
    try:
        with open(file_path, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    timestamp_str = row["Timestamp"].strip()
                    next_epoch_time_str = row["Next Epoch Time"].strip()
                    next_epoch_number = row["Next Epoch"].strip()

                    # Skip empty or bad rows
                    if not timestamp_str or not next_epoch_time_str or not next_epoch_number:
                        continue

                    ts = datetime.strptime(timestamp_str, "%Y-%m-%d %I:%M:%S %p")
                    next_time = datetime.strptime(next_epoch_time_str, "%Y-%m-%d %I:%M:%S %p").time()
                    next_epoch_dt = datetime.combine(ts.date(), next_time)

                    timestamps.append(ts)
                    next_epoch_times.append(next_epoch_dt)
                    next_epoch.append(next_epoch_number)
                except Exception as e:
                    print(f"Skipping row due to error: {e}")
    except FileNotFoundError:
        logging.warning(f"{file_path} not found.")
    return timestamps, next_epoch_times, next_epoch


if __name__ == "__main__":
    logging.info("DB_epoch_timing: JSON files: %s | %s", ROUND_RECORD_FILE, POST_ROUND_RECORD_FILE)
    logging.info("DB_epoch_timing: CSV file: %s", EPOCH_CSV_FILE)
    monitor_json_files()


