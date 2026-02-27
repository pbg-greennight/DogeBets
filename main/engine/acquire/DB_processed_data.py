# DogeBets/main/engine/acquire/DB_processed_data.py
"""
Runtime coordinator for BBFN dominant-trend inference.

v1.04 additions:
- Calls both `ochlv_2ndlast_fetch()` and `ochlv_fetch()` each cycle so the
  exact windows used by the model are logged before inference.
- Exposes the two fetchers as public helpers that `db_trend_calc.py` imports.
- Logs a one-line summary after inference:
    "Final Recorded Trend for Next Epoch: <next_epoch>  =  <LABEL>  (confidence=<x.xxx>)"
- Small cleanup of scheduling / logging; optional verbose OCHLV printing.

This module intentionally keeps the public functions and signatures stable so
`db_trend_calc.dominate_trend()` can import `ochlv_fetch`/`ochlv_2ndlast_fetch`.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
from main.engine.acquire.DB_processed_data_helpers import _pick, _attach_today_est_times, _parse_epoch_time_est, run_continuously
from main.engine.process.DB_DATA_TREND_PROCESS import fetch_last_epoch_info
import numpy as np
import pandas as pd
from main.engine.acquire.DB_processed_data_model import (
    load_merged_csv,
    build_epoch_df,
    add_stage2_features,
    EPOCH_TAIL_EVAL,
    DIRECTION_LABEL_COL,
    _baseline_A_predict,
    _baseline_B_predict,
    _baseline_C_predict,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - INFO - %(message)s",
    datefmt="%I:%M:%S %p",
    force=True,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OCHLV_CSV = BASE_DIR / "csv/DB_OCHLV_DATA.csv"
EPOCH_MERGED_CSV = BASE_DIR / "csv" / "DB_epoch_ochlv_merged.csv"
# Verbose toggle for printing raw O/H/L/C/V inside each window
VERBOSE_OHLCV_LOGS = True

# How many seconds before the official next-epoch time to wake up
WAKEUP_MARGIN_SECONDS = 14
MODEL_RUN_DELAY_SECONDS = 60


def _load_epoch_history() -> Optional[pd.DataFrame]:
    """
    Build epoch_history for Stage 2 / vol_z_20 during live inference.

    Steps:
        - Load merged epoch CSV
        - Build epoch_df (Stage 1 features)
        - Add Stage 2 features (flip-aware additions)
        - Return only the columns used by compute_features_for_live_epoch
    """
    try:
        # 1) Load merged CSV
        merged = load_merged_csv(EPOCH_MERGED_CSV)

        # 2) Build Stage 1 epoch_df
        epoch_df = build_epoch_df(merged)

        # 3) Add Stage 2 features (flip-aware indicators)
        epoch_df = add_stage2_features(epoch_df)

    except Exception as e:
        logging.error(f"❌ Failed to build epoch_history from {EPOCH_MERGED_CSV}: {e}")
        return None

    # Columns that live inference can use
    desired_cols = [
        "endPrice_num",
        "volume_epoch",
        "priceDiff_num",
        "approx_priceDiff_num",
        "direction",
        "open_epoch",
        "close_epoch",
        "high_epoch",
        "low_epoch",
        "run_length_signed",
        "atr_14",
        "bb_width_20",
    ]

    # Required minimum columns
    required_min = ["endPrice_num", "volume_epoch", "priceDiff_num"]
    missing_req = [c for c in required_min if c not in epoch_df.columns]
    if missing_req:
        logging.error(f"❌ epoch_df missing required columns: {missing_req}")
        return None

    # Keep only valid columns
    keep_cols = [c for c in desired_cols if c in epoch_df.columns]

    return epoch_df[keep_cols].copy()


def _resolve_csv_path() -> Path:
    try:
        return OCHLV_CSV
    except NameError:  # fallback if constant moved
        return Path(r"/main/engine/acquire/csv/DB_OCHLV_DATA.csv")

def _load_ochlv_csv() -> Optional[pd.DataFrame]:
    """Load the OCHLV CSV produced by DB_DATA_FETCH.py.

    That script writes a header row:
        Timestamp,Open,High,Low,Close,Volume

    If for any reason the file was created without a header and the first data
    row became the header, we fall back to re-reading with explicit names.
    """
    csv_path = _resolve_csv_path()
    if not Path(csv_path).exists():
        logging.error(f"❌ OCHLV CSV not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"❌ Failed to read OCHLV CSV: {e}")
        return None

    cols = list(df.columns)
    if "Timestamp" not in cols and len(cols) == 6:
        # Likely headerless; re-read with explicit column names
        try:
            df = pd.read_csv(
                csv_path,
                header=None,
                names=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
            )
         #   logging.info("🔧 Re-read OCHLV CSV assuming missing header; applied standard columns.")
        except Exception as e:
            logging.error(f"❌ Fallback re-read of OCHLV CSV failed: {e}")
            return None

    return df

# -----------------------------------------------------------------------------
# OCHLV window fetchers (used by db_trend_calc / inference)
# -----------------------------------------------------------------------------
def current_epoch_trend() -> Optional[pd.DataFrame]:
    """Return a DataFrame slice for the *current* epoch window [curr_start, now].

    Also (optionally) log O/H/L/C/V for that window using NY time.
    """
    df = _load_ochlv_csv()
    if df is None:
        return None
    if df.empty:
        logging.info("ℹ️ OCHLV CSV is empty; nothing to log.")
        return df

    # Resolve columns
    cols = list(df.columns)
    open_col = _pick(cols, "open")
    high_col = _pick(cols, "high")
    low_col = _pick(cols, "low")
    close_col = _pick(cols, "close", "price")
    volume_col = _pick(cols, "volume", "vol")
    time_col = _pick(cols, "timestamp", "time")

    missing = [n for n, c in {
        "Timestamp": time_col,
        "Open": open_col,
        "High": high_col,
        "Low": low_col,
        "Close": close_col,
        "Volume": volume_col,
    }.items() if c is None]
    if missing:
        logging.warning(f"⚠️ Missing expected columns: {', '.join(missing)}")
        logging.warning(f"🔎 Available CSV columns: {', '.join(map(str, cols))}")
        return df

    df = _attach_today_est_times(df, time_col)

    # Epoch bounds
    info = fetch_last_epoch_info()
    (
        prev_epoch,
        prev_epoch_time,
        curr_epoch,
        curr_epoch_time,
        next_epoch,
        next_epoch_time,
    ) = info

    try:
        # Curr epoch start (anchor) is the previous epoch time; we build from
        # the *current* epoch anchor to "now" for the live window.
        start_dt = _parse_epoch_time_est(curr_epoch_time)
        # now (NY-aware)
        import pytz
        now_est = datetime.now(pytz.timezone("America/New_York"))
        end_dt = now_est
    except Exception as e:
        logging.error(f"💥 Error building epoch window: {e}")
        return df

    window = df.loc[(df["__ts"] >= start_dt) & (df["__ts"] <= end_dt)]

    logging.info("-" * 106)
    logging.info(f"OCHLV Window Data: {len(window)} rows from last epoch ({curr_epoch} at {start_dt.strftime('%Y-%m-%d %I:%M:%S %p')}) "
                 f" to  current epoch ({next_epoch} at {end_dt.strftime('%Y-%m-%d %I:%M:%S %p')})")

    if VERBOSE_OHLCV_LOGS and not window.empty:
        times_fmt = window["__ts"].dt.tz_convert("America/New_York").dt.strftime("%I:%M:%S %p")
        open_line = ", ".join(f"{t} : {v:.2f}" for t, v in zip(times_fmt, window[open_col]))
        close_line = ", ".join(f"{t} : {v:.2f}" for t, v in zip(times_fmt, window[close_col]))
        high_line = ", ".join(f"{t} : {v:.2f}" for t, v in zip(times_fmt, window[high_col]))
        low_line = ", ".join(f"{t} : {v:.2f}" for t, v in zip(times_fmt, window[low_col]))
        volume_line = ", ".join(f"{t} : {v:.5g}" for t, v in zip(times_fmt, window[volume_col]))

     #   logging.info(f"-------------------------- This data used to predict trend for epoch {next_epoch} ---------------------------")
     #   logging.info(f"Open   =  {open_line}")
     #   logging.info(f"Close  =  {close_line}")
     #   logging.info(f"High   =  {high_line}")
     #   logging.info(f"Low    =  {low_line}")
     #   logging.info(f"Volume =  {volume_line}")

    return window.copy()


def infer_trend_for_next_epoch(
    epoch_block: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """
    Decide the trend for the *next* epoch using ONLY the rule-based baselines.

    This function no longer uses the AuSTD tree model for live trading.
    Instead it:

      1. Rebuilds the full epoch_df from the merged CSV.
      2. Adds Stage 2 indicators (EMA, MACD, RSI, ATR, Bollinger bands).
      3. Looks at the last EPOCH_TAIL_EVAL epochs and measures how well
         Baseline A/B/C would have matched the true labels on that tail.
      4. Picks the baseline with the highest tail accuracy.
      5. Applies that winning baseline to the most recent epoch row and
         returns a 3-class decision: Bear(0), Neutral(1), Bull(2).

    Notes
    -----
    * RF/GB models are still used in the offline trainer to learn the
      environment, but **they do not participate in live decisions** here.
    * This mirrors the conceptual pipeline:
          - use trees to learn,
          - use baselines to test recent regime,
          - let the best baseline call n+1.
    """
    # 1) Rebuild epoch_df from the merged CSV
    try:
        merged = load_merged_csv(EPOCH_MERGED_CSV)
    except Exception as e:
        logging.warning(
            "⚠️ Failed to load merged CSV for live baseline inference: %s", e
        )
        return None

    try:
        epoch_df = build_epoch_df(merged)
    except Exception as e:
        logging.warning(
            "⚠️ Failed to build epoch_df for live baseline inference: %s", e
        )
        return None

    if epoch_df.empty or len(epoch_df) < max(3, EPOCH_TAIL_EVAL):
        logging.warning(
            "⚠️ epoch_df too small for baseline inference "
            "(len=%d, tail=%d); skipping.",
            len(epoch_df),
            EPOCH_TAIL_EVAL,
        )
        return None

    # 2) Add Stage 2 indicators so baselines A/B/C have what they need
    try:
        epoch_df = add_stage2_features(epoch_df)
    except Exception as e:
        logging.warning(
            "⚠️ Failed to add Stage 2 features for live baseline inference: %s", e
        )
        return None

    # 3) Build tail window for baseline accuracy measurement
    tail_df = epoch_df.tail(EPOCH_TAIL_EVAL).copy()

    # Columns we need for tail evaluation
    required_cols = [
        DIRECTION_LABEL_COL,   # true 3-class label, "direction_3"
        "Close",
        "Open",
        "High",
        "Low",
        "rsi_14",
        "atr_14",
        "macd_hist",
        "slope_C_10",
        "pctB_20",
        "bb_width_20",
    ]
    missing = [c for c in required_cols if c not in tail_df.columns]
    if missing:
        logging.warning(
            "⚠️ Tail baseline inference missing required columns %s; "
            "cannot evaluate baselines; skipping.",
            missing,
        )
        return None

    # Clean NaNs in the tail
    tail_df = tail_df.dropna(subset=required_cols)
    if tail_df.empty:
        logging.warning(
            "⚠️ Tail baseline inference has no valid rows after cleanup; skipping."
        )
        return None

    from sklearn.metrics import accuracy_score

    # True labels for the tail
    y_true = tail_df[DIRECTION_LABEL_COL].astype(int).to_numpy()

    # 4) Run all three baselines on the tail
    y_A = _baseline_A_predict(tail_df)
    y_B = _baseline_B_predict(tail_df)
    y_C = _baseline_C_predict(tail_df)

    acc_A = float(accuracy_score(y_true, y_A)) if len(y_true) else 0.0
    acc_B = float(accuracy_score(y_true, y_B)) if len(y_true) else 0.0
    acc_C = float(accuracy_score(y_true, y_C)) if len(y_true) else 0.0

    engines = {
        "baseline_A": acc_A,
        "baseline_B": acc_B,
        "baseline_C": acc_C,
    }
    winner = max(engines, key=engines.get)
    acc_winner = engines[winner]

    logging.info(
        "🧪 Live-tail baselines (last %d epochs): "
        "A=%.4f, B=%.4f, C=%.4f -> winner=%s",
        len(tail_df),
        acc_A,
        acc_B,
        acc_C,
        winner,
    )

    # 5) Apply winning baseline to the most recent context
    # Give baselines the last 2 rows so any simple diffs still work
    live_ctx = epoch_df.tail(2).copy()
    if live_ctx.empty:
        logging.warning("⚠️ No rows in live_ctx for baseline inference; skipping.")
        return None

    if winner == "baseline_A":
        live_preds = _baseline_A_predict(live_ctx)
    elif winner == "baseline_B":
        live_preds = _baseline_B_predict(live_ctx)
    else:
        live_preds = _baseline_C_predict(live_ctx)

    # The last prediction is our call for "next epoch" direction
    live_label = int(live_preds[-1])

    # Synthesize simple pseudo-probabilities for logging / downstream use
    # 0 = Bear, 1 = Neutral, 2 = Bull
    if live_label == 0:
        P_bear, P_neutral, P_bull = 0.90, 0.05, 0.05
    elif live_label == 2:
        P_bear, P_neutral, P_bull = 0.05, 0.05, 0.90
    else:
        P_bear, P_neutral, P_bull = 0.10, 0.80, 0.10

    return {
        "label": live_label,
        "P_bear": P_bear,
        "P_bull": P_bull,
        "P_neutral": P_neutral,
        "engine": winner,
        "engine_scores": engines,
        "engine_acc": acc_winner,
    }


def _log_epoch_header(info: Tuple[Any, ...]) -> None:
    (   prev_epoch,
        prev_epoch_time,
        curr_epoch,
        curr_epoch_time,
        next_epoch,
        next_epoch_time,
    ) = info

 #   logging.info(f" 2nd to last Epoch:  {prev_epoch},  2nd to last Epoch Time:  {prev_epoch_time}")
 #   logging.info(f" Previous Epoch:     {curr_epoch},  Previous Epoch Time:     {curr_epoch_time}")
 #   logging.info(f" Current Epoch:      {next_epoch},  Current Epoch Time:      {next_epoch_time}")


__all__ = [
    "run_continuously",
]

if __name__ == "__main__":
    run_continuously()
