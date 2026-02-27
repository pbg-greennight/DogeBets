# DB_epoch_ochlv_merged.py

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


# =============================================================================
# Paths / Config
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Input CSVs
OCHLV_FILE = BASE_DIR / "csv/DB_OCHLV_DATA.csv"
EPOCH_FILE = BASE_DIR / "csv/DB_epoch_time.csv"

# Output merged CSV
MERGED_FILE = BASE_DIR / "csv/DB_epoch_ochlv_merged.csv"

# This must match WAKEUP_MARGIN_SECONDS in DB_processed_data.py
WAKEUP_MARGIN_SECONDS = 18

@dataclass
class PathsConfig:
    ochlv_csv: Path
    epoch_csv: Path
    merged_csv: Path


@dataclass
class RuntimeConfig:
    poll_seconds: float = 3.0  # live loop sleep interval


class RunMode(Enum):
    INITIAL_BACKFILL = auto()
    RECOVERY_BACKFILL = auto()
    LIVE_ONLY = auto()


# =============================================================================
# Logging
# =============================================================================

def setup_logging() -> None:
    """Configure simple timestamped logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# =============================================================================
# Timestamp parsing
# =============================================================================

def parse_timestamp(value: str) -> Optional[datetime]:
    """
    Parse timestamp strings used in both CSVs.

    Known patterns from your files:
    - OCHLV: "2025-11-20 09:12:05 PM"
    - Epoch Timestamp: "11/11/2025 11:04"
    - Next Epoch Time: "11/20/2025  09:12:43 PM"
    """
    if not isinstance(value, str):
        return None

    # Collapse any weird spacing (e.g. double spaces between date/time)
    value = " ".join(value.strip().split())
    if not value:
        return None

    fmts: List[str] = [
        "%Y-%m-%d %I:%M:%S %p",  # 2025-11-20 09:12:05 PM
        "%m/%d/%Y %I:%M:%S %p",  # 11/20/2025 09:12:43 PM
        "%m/%d/%Y %I:%M %p",     # 11/20/2025 09:12 PM
        "%m/%d/%Y %H:%M:%S",     # 11/20/2025 12:56:03
        "%m/%d/%Y %H:%M",        # 11/11/2025 11:04
    ]

    for fmt in fmts:
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            continue

    return None


# =============================================================================
# OCHLV loading / normalization
# =============================================================================

def _read_ochlv_raw(path: Path) -> pd.DataFrame:
    """
    Read the raw OCHLV CSV, handling the fact that your file often has no header.

    If the first column name looks like a datetime string, we assume there is
    NO header and reload with header=None.
    """
    if not path.exists():
        logging.warning("OCHLV file does not exist: %s", path)
        return pd.DataFrame()

    # First attempt: default read
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.error("Failed to read OCHLV CSV %s: %s", path, e)
        return pd.DataFrame()

    if df.empty:
        return df

    first_col_name = str(df.columns[0])
    # If the column name itself parses as a timestamp, treat file as headerless
    if parse_timestamp(first_col_name) is not None:
        # Re-read with no header and assign standard names
        try:
            df = pd.read_csv(
                path,
                header=None,
                names=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
            )
        except Exception as e:
            logging.error("Failed to re-read OCHLV CSV as headerless: %s", e)
            return pd.DataFrame()
        return df

    # Else, return as-is (header present)
    return df


def normalize_ochlv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure standard OCHLV column names.

    Supports:
    - Timestamp, Open, High, Low, Close, Volume
    - Timestamp, BTC_Open, BTC_High, BTC_Low, BTC_Close, BTC_Volume
    """
    if "BTC_Open" in df.columns:
        df = df.rename(columns={
            "BTC_Open": "Open",
            "BTC_High": "High",
            "BTC_Low": "Low",
            "BTC_Close": "Close",
            "BTC_Volume": "Volume",
        })
    return df


def load_ochlv_df(path: Path) -> pd.DataFrame:
    """
    Load OCHLV CSV, normalize, parse timestamps, and sort.

    We keep ALL rows (no dedupe) to match your raw file exactly.
    """
    df = _read_ochlv_raw(path)
    if df.empty:
        return df

    df = normalize_ochlv_columns(df)

    required = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            logging.warning("OCHLV missing column: %s", col)
            return pd.DataFrame()

    # Parse timestamps
    df["ts"] = df["Timestamp"].apply(parse_timestamp)
    before = len(df)
    df = df.dropna(subset=["ts"])
    dropped = before - len(df)
    if dropped:
        logging.warning("Dropped %d OCHLV rows due to bad timestamps", dropped)

    # Convert numeric fields (keep rows that convert cleanly)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    dropped = before - len(df)
    if dropped:
        logging.warning("Dropped %d OCHLV rows due to non-numeric price fields", dropped)

    df = df.sort_values("ts").reset_index(drop=True)
    return df


# =============================================================================
# Epoch loading
# =============================================================================

def load_epoch_df(path: Path) -> pd.DataFrame:
    """
    Load epoch CSV, parse 'Timestamp' into 'ts', sort.

    We leave startPrice/endPrice/priceDifference as strings for now so their
    "$" and parentheses formatting is preserved in the merged output.
    """
    if not path.exists():
        logging.warning("Epoch file does not exist: %s", path)
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.error("Failed to read epoch CSV %s: %s", path, e)
        return pd.DataFrame()

    if "Timestamp" not in df.columns:
        logging.warning("Epoch CSV missing 'Timestamp' column.")
        return pd.DataFrame()

    df["ts"] = df["Timestamp"].apply(parse_timestamp)
    before = len(df)
    df = df.dropna(subset=["ts"])
    dropped = before - len(df)
    if dropped:
        logging.warning("Dropped %d epoch rows due to bad timestamps", dropped)

    # Numeric columns we might use later; safe conversion now
    for col in ["Current Epoch", "Next Epoch", "previousEpoch"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("ts").reset_index(drop=True)
    return df


# =============================================================================
# Merged CSV schema / helpers
# =============================================================================

def get_merged_columns() -> list[str]:
    return [
        "Timestamp", "Open", "High", "Low", "Close", "Volume",
        "Current Epoch", "Next Epoch", "Next Epoch Time",
        "previousEpoch", "startPrice", "endPrice", "priceDifference",
        "approx_endPrice", "approx_priceDifference",
    ]


def ensure_merged_csv_header(paths: PathsConfig) -> None:
    """
    Ensure merged CSV exists and has header.

    If file does not exist, create it with header only.
    If it exists, we assume header is already correct.
    """
    if not paths.merged_csv.exists():
        paths.merged_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = get_merged_columns()
        pd.DataFrame(columns=cols).to_csv(paths.merged_csv, index=False)
        logging.info("Created merged CSV with header: %s", paths.merged_csv)


def read_last_merged_timestamp(paths: PathsConfig) -> Optional[datetime]:
    """
    Read last 'Timestamp' from merged CSV. Returns None if no data rows.
    """
    if not paths.merged_csv.exists():
        return None

    try:
        df = pd.read_csv(paths.merged_csv)
    except Exception as e:
        logging.error("Failed to read merged CSV: %s", e)
        return None

    if df.empty:
        return None

    # parse last timestamp
    last_ts_str = str(df["Timestamp"].iloc[-1])
    return parse_timestamp(last_ts_str)


# =============================================================================
# Core mapping: OCHLV → Epoch
# =============================================================================

def map_ochlv_to_epoch(ochlv_df: pd.DataFrame, epoch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach epoch context to OCHLV rows using Next Epoch Time and closest neighbor.
    """

    merged_cols = get_merged_columns()

    # If no OCHLV data, nothing to do
    if ochlv_df.empty:
        return pd.DataFrame(columns=merged_cols)

    # Start with OCHLV copy and ensure epoch columns exist and are NaN (as object)
    out = ochlv_df.sort_values("ts").reset_index(drop=True).copy()

    epoch_cols = [
        "Current Epoch", "Next Epoch", "previousEpoch",
        "startPrice", "endPrice", "priceDifference",
    ]

    # Make sure these columns exist as object dtype (so strings are fine)
    if "Next Epoch Time" not in out.columns:
        out["Next Epoch Time"] = pd.Series([np.nan] * len(out), dtype="object")
    else:
        out["Next Epoch Time"] = out["Next Epoch Time"].astype("object")

    for col in epoch_cols:
        if col not in out.columns:
            out[col] = pd.Series([np.nan] * len(out), dtype="object")
        else:
            out[col] = out[col].astype("object")

    # If no epoch data, just return OCHLV-only merged schema
    if epoch_df.empty:
        for col in merged_cols:
            if col not in out.columns:
                out[col] = np.nan
        return out[merged_cols]

    # Work on a copy of epoch_df to compute next_ts
    ep = epoch_df.copy()

    if "Next Epoch Time" in ep.columns:
        ep["next_ts"] = ep["Next Epoch Time"].apply(parse_timestamp)
    else:
        ep["next_ts"] = pd.NaT

    ep_valid = ep[ep["next_ts"].notna()].copy()
    if ep_valid.empty:
        for col in merged_cols:
            if col not in out.columns:
                out[col] = np.nan
        return out[merged_cols]

    # Prepare list of OCHLV timestamps for nearest-neighbor search
    och_ts = list(out["ts"])
    n_och = len(och_ts)

    import bisect

    def find_nearest_index(target: datetime) -> Optional[int]:
        """Return index of OCHLV row whose ts is closest to target."""
        if n_och == 0:
            return None
        pos = bisect.bisect_left(och_ts, target)
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < n_och:
            candidates.append(pos)
        if not candidates:
            return None
        best_idx = None
        best_diff = None
        for idx in candidates:
            diff = abs((och_ts[idx] - target).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = idx
        return best_idx

    # Stamp epoch data into nearest OCHLV row (once per epoch row)
    for _, er in ep_valid.iterrows():
        target = er["next_ts"]
        idx = find_nearest_index(target)
        if idx is None:
            continue

        for col in epoch_cols:
            if col in er:
                out.at[idx, col] = er[col]

        # Next Epoch Time = OCHLV Timestamp string (for visual match)
        out.at[idx, "Next Epoch Time"] = out.at[idx, "Timestamp"]

    # Ensure final column order and presence
    for col in merged_cols:
        if col not in out.columns:
            out[col] = np.nan

    return out[merged_cols]

from datetime import timedelta  # already imported datetime above, add timedelta

def add_approx_price_diff(
    merged: pd.DataFrame,
    wakeup_margin_seconds: int = WAKEUP_MARGIN_SECONDS,
) -> pd.DataFrame:
    """
    Add approx_endPrice / approx_priceDifference columns to the merged OCHLV+epoch CSV.

    For each epoch anchor row (where startPrice/endPrice/priceDifference are present),
    we approximate the end-of-epoch close as the last OCHLV Close in the window:

        (prev_boundary, boundary - wakeup_margin_seconds]

    where:
      - boundary       = Timestamp of the current epoch anchor row
      - prev_boundary  = Timestamp of the previous epoch anchor row

    This uses ONLY data that would have been known at live decision time, i.e.
    WAKEUP_MARGIN_SECONDS before the epoch boundary — no future peeking.
    """
    if merged.empty:
        return merged

    df = merged.copy()

    # Parse OCHLV timestamps
    df["_ts"] = df["Timestamp"].apply(parse_timestamp)

    # Numeric Close
    df["_close_num"] = pd.to_numeric(df["Close"], errors="coerce")

    # Parse numeric startPrice (strip $, commas, parentheses)
    start_str = df["startPrice"].astype(str)
    start_clean = start_str.str.replace(r"[\$,()]", "", regex=True).replace("", np.nan)
    df["_start_num"] = pd.to_numeric(start_clean, errors="coerce")

    # Ensure approx columns exist
    if "approx_endPrice" not in df.columns:
        df["approx_endPrice"] = np.nan
    if "approx_priceDifference" not in df.columns:
        df["approx_priceDifference"] = np.nan

    # Identify epoch anchor rows: where priceDifference & startPrice are present
    is_anchor = df["priceDifference"].notna() & df["_start_num"].notna() & df["_ts"].notna()
    anchor_idx = df.index[is_anchor].tolist()

    prev_boundary_ts: Optional[datetime] = None

    for idx in anchor_idx:
        boundary_ts = df.at[idx, "_ts"]
        if prev_boundary_ts is None:
            # No previous boundary -> can't define a clean epoch window; skip first anchor.
            prev_boundary_ts = boundary_ts
            continue

        call_time = boundary_ts - timedelta(seconds=wakeup_margin_seconds)

        # All OCHLV ticks in (prev_boundary, call_time]
        mask_seg = (df["_ts"] > prev_boundary_ts) & (df["_ts"] <= call_time)
        seg = df.loc[mask_seg]

        approx_close = np.nan
        if not seg.empty:
            last_idx = seg.index[-1]
            approx_close = df.at[last_idx, "_close_num"]

        if not pd.isna(approx_close):
            start_num = df.at[idx, "_start_num"]
            if not pd.isna(start_num):
                df.at[idx, "approx_endPrice"] = approx_close
                df.at[idx, "approx_priceDifference"] = approx_close - start_num

        # Move boundary forward for next epoch
        prev_boundary_ts = boundary_ts

    # Drop helper columns
    df = df.drop(columns=["_ts", "_close_num", "_start_num"], errors="ignore")
    return df

# =============================================================================
# Backfill operations
# =============================================================================

def run_initial_backfill(paths: PathsConfig) -> Optional[datetime]:
    """
    Build merged CSV from scratch (full history).
    """
    logging.info("Running initial backfill...")

    ochlv_df = load_ochlv_df(paths.ochlv_csv)
    epoch_df = load_epoch_df(paths.epoch_csv)

    logging.info("Loaded OCHLV rows: %d", len(ochlv_df))
    logging.info("Loaded epoch rows: %d", len(epoch_df))

    if ochlv_df.empty:
        logging.warning("No OCHLV data for initial backfill; nothing to do.")
        return None

    # Limit epoch rows to those whose Next Epoch Time is within the OCHLV window
    if not epoch_df.empty and "Next Epoch Time" in epoch_df.columns:
        tmp = epoch_df.copy()
        tmp["next_ts"] = tmp["Next Epoch Time"].apply(parse_timestamp)
        max_och_ts = ochlv_df["ts"].max()
        tmp = tmp[tmp["next_ts"].notna() & (tmp["next_ts"] <= max_och_ts)]
        epoch_df = tmp
        logging.info("Initial backfill epoch rows after next_ts filter: %d", len(epoch_df))

    merged = map_ochlv_to_epoch(ochlv_df, epoch_df)
    if merged.empty:
        logging.warning("Merged data is empty after initial backfill.")
        return None

    # 🔧 NEW: compute approx_endPrice / approx_priceDifference
    merged = add_approx_price_diff(merged, wakeup_margin_seconds=WAKEUP_MARGIN_SECONDS)

    merged.to_csv(paths.merged_csv, index=False)
    last_ts = parse_timestamp(str(merged["Timestamp"].iloc[-1]))
    logging.info("Initial backfill wrote %d rows. Last ts=%s", len(merged), last_ts)
    return last_ts


def run_recovery_backfill(paths: PathsConfig, last_merged_ts: datetime) -> Optional[datetime]:
    """
    Append missing rows between last_merged_ts and newest OCHLV rows.
    """
    logging.info("Running recovery backfill from %s ...", last_merged_ts)

    ochlv_df = load_ochlv_df(paths.ochlv_csv)
    epoch_df = load_epoch_df(paths.epoch_csv)

    if ochlv_df.empty:
        logging.warning("No OCHLV data available; skipping recovery backfill.")
        return last_merged_ts

    # Filter OCHLV strictly after last merged timestamp
    tail = ochlv_df[ochlv_df["ts"] > last_merged_ts].copy()
    if tail.empty:
        logging.info("No new OCHLV rows after last merged timestamp; nothing to backfill.")
        return last_merged_ts

    merged_tail = map_ochlv_to_epoch(tail, epoch_df)
    if merged_tail.empty:
        logging.warning("Mapped tail is empty; not appending anything.")
        return last_merged_ts

    # 🔧 NEW: compute approx_* for the tail chunk as well
    merged_tail = add_approx_price_diff(merged_tail, wakeup_margin_seconds=WAKEUP_MARGIN_SECONDS)

    # Append (no header)
    merged_tail.to_csv(paths.merged_csv, mode="a", header=False, index=False)

    # Append (no header)
    merged_tail.to_csv(paths.merged_csv, mode="a", header=False, index=False)
    new_last_ts = parse_timestamp(str(merged_tail["Timestamp"].iloc[-1]))
    logging.info(
        "Recovery backfill appended %d rows. New last ts=%s",
        len(merged_tail),
        new_last_ts,
    )
    return new_last_ts or last_merged_ts


# =============================================================================
# Live loop
# =============================================================================

def run_live_loop(paths: PathsConfig, runtime: RuntimeConfig, last_merged_ts: Optional[datetime]) -> None:
    """
    Live loop:
    - On each iteration, load OCHLV/epoch
    - Take OCHLV rows with ts > last_merged_ts
    - Map & append, using only epochs whose Next Epoch Time is
      between (last_merged_ts, max_new_och_ts].
    """
    logging.info("Starting live loop... poll=%.1fs", runtime.poll_seconds)

    while True:
        try:
            ochlv_df = load_ochlv_df(paths.ochlv_csv)
            epoch_df_full = load_epoch_df(paths.epoch_csv)

            if ochlv_df.empty:
                logging.info("OCHLV empty; sleeping %.1fs", runtime.poll_seconds)
                time.sleep(runtime.poll_seconds)
                continue

            # 1) Find new OCHLV rows since last_merged_ts
            if last_merged_ts is not None:
                new_rows = ochlv_df[ochlv_df["ts"] > last_merged_ts].copy()
            else:
                new_rows = ochlv_df.copy()

            if new_rows.empty:
                time.sleep(runtime.poll_seconds)
                continue

            max_new_och_ts = new_rows["ts"].max()

            # 2) Restrict epoch rows to the current OCHLV window
            #    We want epochs whose Next Epoch Time is between
            #    (last_merged_ts, max_new_och_ts].
            if (
                not epoch_df_full.empty
                and "Next Epoch Time" in epoch_df_full.columns
            ):
                tmp = epoch_df_full.copy()
                tmp["next_ts"] = tmp["Next Epoch Time"].apply(parse_timestamp)

                if last_merged_ts is not None:
                    mask = (
                        tmp["next_ts"].notna()
                        & (tmp["next_ts"] > last_merged_ts)
                        & (tmp["next_ts"] <= max_new_och_ts)
                    )
                else:
                    # If we somehow start live loop with no last_merged_ts,
                    # just use epochs up to the current max_new_och_ts.
                    mask = tmp["next_ts"].notna() & (tmp["next_ts"] <= max_new_och_ts)

                epoch_df_live = tmp[mask]
            else:
                epoch_df_live = epoch_df_full

            # 3) Map only the new rows, with the epochs in this window
            merged_new = map_ochlv_to_epoch(new_rows, epoch_df_live)

            if merged_new.empty:
                # No mapping; still advance last_merged_ts so we don't
                # re-process these OCHLV rows again.
                last_merged_ts = max_new_och_ts
                time.sleep(runtime.poll_seconds)
                continue

            # 4) Append merged chunk to CSV
            merged_new.to_csv(paths.merged_csv, mode="a", header=False, index=False)
            last_merged_ts = parse_timestamp(str(merged_new["Timestamp"].iloc[-1]))

            logging.info(
                "Live loop appended %d rows. last_merged_ts=%s",
                len(merged_new),
                last_merged_ts,
            )

            # 5) Backfill any epoch details that just became available
            run_live_backfill(paths)

            time.sleep(runtime.poll_seconds)


        except KeyboardInterrupt:
            logging.info("Live loop interrupted by user. Exiting.")
            break
        except Exception as e:
            logging.exception("Error in live loop: %s", e)
            time.sleep(runtime.poll_seconds)

def run_live_backfill(paths: PathsConfig) -> None:
    """
    Live backfill of delayed epoch details.

    During live operation, DB_epoch_time.csv will only get full values for
    previousEpoch, startPrice, endPrice, priceDifference a couple of epochs
    after the epoch actually happened.

    This function:
      - Reads the latest epoch CSV.
      - Finds epoch rows where those fields are now populated.
      - Opens the merged CSV and, for each matching Current Epoch, fills in
        any missing previousEpoch/startPrice/endPrice/priceDifference values.
    """
    # 1) Load latest epoch data
    epoch_df = load_epoch_df(paths.epoch_csv)
    if epoch_df.empty:
        return

    # We only care about rows where any of these are now non-null
    detail_cols = ["previousEpoch", "startPrice", "endPrice", "priceDifference"]
    available_mask = False
    for col in detail_cols:
        if col in epoch_df.columns:
            available_mask = available_mask | epoch_df[col].notna()

    if not isinstance(available_mask, pd.Series) or not available_mask.any():
        # No rows with any detail filled; nothing to backfill
        return

    epoch_with_details = epoch_df[available_mask].copy()

    # 2) Load merged CSV
    if not paths.merged_csv.exists():
        return

    try:
        merged_df = pd.read_csv(paths.merged_csv)
    except Exception as e:
        logging.error("run_live_backfill: failed to read merged CSV: %s", e)
        return

    if merged_df.empty:
        return

    # Ensure the epoch detail columns exist and are object dtype in merged_df
    for col in detail_cols:
        if col not in merged_df.columns:
            merged_df[col] = pd.Series([np.nan] * len(merged_df), dtype="object")
        else:
            merged_df[col] = merged_df[col].astype("object")

    # For matching, coerce merged_df["Current Epoch"] to numeric, independent of dtype
    if "Current Epoch" not in merged_df.columns:
        return

    merged_cur_epoch = pd.to_numeric(merged_df["Current Epoch"], errors="coerce")

    # 3) For each epoch row that now has details, patch the merged rows
    updated_rows = 0

    for _, er in epoch_with_details.iterrows():
        cur_epoch = er.get("Current Epoch", np.nan)
        if pd.isna(cur_epoch):
            continue

        # rows in merged_df that correspond to this Current Epoch
        mask_epoch = merged_cur_epoch == cur_epoch
        if not mask_epoch.any():
            continue

        # For each detail column, only fill where merged_df is still NaN/empty
        for col in detail_cols:
            if col not in er or col not in merged_df.columns:
                continue

            value = er[col]
            if pd.isna(value):
                continue

            def needs_update(x):
                if pd.isna(x):
                    return True
                if isinstance(x, str) and x.strip() == "":
                    return True
                return False

            # indices where we want to write this value
            idx_to_update = merged_df.index[mask_epoch & merged_df[col].map(needs_update)]
            if len(idx_to_update) == 0:
                continue

            merged_df.loc[idx_to_update, col] = value
            updated_rows += len(idx_to_update)

    if updated_rows > 0:
        # 4) Write merged CSV back to disk
        try:
            merged_df.to_csv(paths.merged_csv, index=False)
            logging.info(
                "run_live_backfill: updated %d cell(s) of epoch detail data.",
                updated_rows,
            )
        except Exception as e:
            logging.error("run_live_backfill: failed to write merged CSV: %s", e)

# =============================================================================
# Run mode detection & main
# =============================================================================

def detect_run_mode(paths: PathsConfig) -> RunMode:
    """
    Decide INITIAL_BACKFILL, RECOVERY_BACKFILL, or LIVE_ONLY.
    """
    if not paths.merged_csv.exists():
        logging.info("Merged CSV does not exist; selecting INITIAL_BACKFILL.")
        return RunMode.INITIAL_BACKFILL

    try:
        df = pd.read_csv(paths.merged_csv)
    except Exception as e:
        logging.error("Failed to read merged CSV to detect run mode: %s", e)
        return RunMode.INITIAL_BACKFILL

    if df.empty:
        logging.info("Merged CSV has header but no data; selecting INITIAL_BACKFILL.")
        return RunMode.INITIAL_BACKFILL

    # Compare last merged TS vs latest OCHLV TS
    last_merged_ts = parse_timestamp(str(df["Timestamp"].iloc[-1]))

    ochlv_df = load_ochlv_df(paths.ochlv_csv)
    if ochlv_df.empty or last_merged_ts is None:
        logging.info("Cannot compare timestamps; selecting LIVE_ONLY.")
        return RunMode.LIVE_ONLY

    latest_ochlv_ts = ochlv_df["ts"].max()

    if last_merged_ts < latest_ochlv_ts:
        logging.info(
            "Merged CSV is behind OCHLV (merged=%s < OCHLV=%s); "
            "selecting RECOVERY_BACKFILL.",
            last_merged_ts,
            latest_ochlv_ts,
        )
        return RunMode.RECOVERY_BACKFILL

    logging.info("Merged CSV is up-to-date; selecting LIVE_ONLY.")
    return RunMode.LIVE_ONLY


def main() -> None:
    setup_logging()

    paths = PathsConfig(
        ochlv_csv=OCHLV_FILE,
        epoch_csv=EPOCH_FILE,
        merged_csv=MERGED_FILE,
    )
    runtime = RuntimeConfig(poll_seconds=5.0)

    ensure_merged_csv_header(paths)

    mode = detect_run_mode(paths)
    last_ts: Optional[datetime] = None

    if mode == RunMode.INITIAL_BACKFILL:
        last_ts = run_initial_backfill(paths)

    elif mode == RunMode.RECOVERY_BACKFILL:
        last_ts = read_last_merged_timestamp(paths)
        if last_ts is not None:
            last_ts = run_recovery_backfill(paths, last_ts)
        else:
            # Fallback: if we can't read last timestamp, redo full backfill
            last_ts = run_initial_backfill(paths)

    elif mode == RunMode.LIVE_ONLY:
        last_ts = read_last_merged_timestamp(paths)

    # After backfill/recovery, drop into live loop
    run_live_loop(paths, runtime, last_ts)


if __name__ == "__main__":
    main()
