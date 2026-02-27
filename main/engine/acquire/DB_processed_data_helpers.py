# DogeBets/main/engine/acquire/DB_processed_data_helpers.py

from datetime import datetime
from typing import Optional, List, Any
import pandas as pd
import logging
import time
from main.engine.DB_round_fetch import fetch_last_epoch_info

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _pick(cols: List[str], *candidates: str) -> Optional[str]:
    """Pick first matching column name ignoring case / underscores."""
    norm = {c.lower().replace("_", ""): c for c in cols}
    for cand in candidates:
        key = cand.lower().replace("_", "")
        if key in norm:
            return norm[key]
    return None

def _attach_today_est_times(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Attach a timezone-aware timestamp column `__ts` using the full
    Timestamp string from DB_OCHLV_DATA.csv.

    DB_DATA_FETCH.py writes timestamps in the form:
        "YYYY-MM-DD HH:MM:SS AM/PM"

    We parse that directly and localize to America/New_York.
    """
    import pytz

    est = pytz.timezone("America/New_York")

    ts_full = pd.to_datetime(
        df[time_col].astype(str).str.strip(),
        format="%Y-%m-%d %I:%M:%S %p",
        errors="coerce",
    )
    ts_full = ts_full.dt.tz_localize(est, nonexistent="NaT", ambiguous="NaT")
    return df.assign(__ts=ts_full).dropna(subset=["__ts"])  # keep only parsable rows

def _parse_epoch_time_est(time_str: str) -> datetime:
    """Parse an epoch timestamp string from DB_epoch_time.csv in EST.

    DB_epoch_time.csv now stores full date+time strings like:
        "MM/DD/YYYY  HH:MM:SS AM"

    We normalize internal spacing and parse with the matching format,
    then localize to America/New_York.
    """
    import pytz

    est = pytz.timezone("America/New_York")

    # Normalize multiple spaces: "11/25/2025  08:40:34 AM" -> "11/25/2025 08:40:34 AM"
    cleaned = " ".join(str(time_str).split())

    dt_naive = datetime.strptime(cleaned, "%m/%d/%Y %I:%M:%S %p")
    dt = est.localize(dt_naive)
    return dt


def run_continuously() -> None:
    """Main loop: align to next-epoch time, log windows, run inference, repeat."""
    # Import from DB_processed_data *inside* the function to avoid circular import
    from main.engine.acquire.DB_processed_data import (
        WAKEUP_MARGIN_SECONDS,
        current_epoch_trend,
        _log_epoch_header,
        infer_trend_for_next_epoch,
    )

    # If next-epoch is ridiculously far away, don't spin forever in the pre-sleep loop.
    # Treat that as "schedule looks stale, re-fetch in a bit."
    MAX_PRE_SLEEP_SECONDS = 355
    NEXT_TOO_FAR_SECONDS = 400

    while True:
        info = fetch_last_epoch_info()
        if not isinstance(info, tuple) or len(info) != 6:
            logging.warning("⚠️ fetch_last_epoch_info() did not return 6 fields; retrying in 45s.")
            time.sleep(45)
            continue

        (
            prev_epoch,
            prev_epoch_time,
            curr_epoch,
            curr_epoch_time,
            next_epoch,
            next_epoch_time,
        ) = info

        # ------------------------------------------------------------------
        # Once per epoch: run AuSTD training using full CSV.
        #
        # We still *do* the training 60s-ish after the epoch, but we call it
        # with quiet=True so that the heavy training logs do not spam the
        # live runtime console. Only WARN/ERROR will show up.
        # ------------------------------------------------------------------
        try:
            if curr_epoch is not None and str(curr_epoch).isdigit():
                # Only train once per epoch
                global _LAST_TRAINED_EPOCH
                if "_LAST_TRAINED_EPOCH" not in globals():
                    _LAST_TRAINED_EPOCH = None

                if _LAST_TRAINED_EPOCH != int(curr_epoch):
                    from main.engine.acquire.DB_processed_data_model import run_AuSTD_training_once

                    try:
                        # Quiet = True: suppress INFO-level training logs
                        run_AuSTD_training_once(quiet=True)
                    except Exception as train_err:
                        logging.error("💥 Error during AuSTD training pass: %s", train_err)
                    else:
                        _LAST_TRAINED_EPOCH = int(curr_epoch)
        except Exception as e:
            logging.error("💥 Error in training scheduler block: %s", e)

        if not next_epoch or next_epoch_time == "N/A":
            logging.warning("⚠️ Next epoch info unavailable. Retrying in 45s.")
            time.sleep(45)
            continue

        # Sleep until the pre-margin wakeup before the next-epoch clock
        while True:
            now = datetime.now()
            try:
                # Epoch times from DB_epoch_time.csv look like:
                #   "MM/DD/YYYY  HH:MM:SS AM"
                # Normalize whitespace first so double-spaces don't break parsing.
                clean_curr = " ".join(str(curr_epoch_time).split())
                clean_next = " ".join(str(next_epoch_time).split())

                curr_dt = datetime.strptime(clean_curr, "%m/%d/%Y %I:%M:%S %p")
                next_dt = datetime.strptime(clean_next, "%m/%d/%Y %I:%M:%S %p")

                seconds_until = (next_dt - now).total_seconds()
                target_sleep = max(0.0, seconds_until - WAKEUP_MARGIN_SECONDS)

                # If next epoch is *way* off in the future, don't spin here forever.
                if seconds_until > NEXT_TOO_FAR_SECONDS:
                    logging.warning(
                        f"⚠️ Next epoch appears {seconds_until:.1f}s away "
                        f"(>{NEXT_TOO_FAR_SECONDS}s). Epoch schedule may be stale; "
                        "sleeping 60s then re-fetching epoch info."
                    )
                    time.sleep(60)
                    # Break out of inner loop so the outer loop can re-call fetch_last_epoch_info()
                    break

                if target_sleep > MAX_PRE_SLEEP_SECONDS:
                    logging.info(
                        f"⏳ Sleep time ({target_sleep:.1f}s) exceeds {MAX_PRE_SLEEP_SECONDS}s. "
                        "Sleeping 25s then rechecking."
                    )
                    time.sleep(25)
                elif target_sleep > 0:
                    logging.info(
                        f"⏳ Sleeping {target_sleep:.1f}s until {WAKEUP_MARGIN_SECONDS}s before next epoch."
                    )
                    logging.info("- - " * 25)
                    time.sleep(target_sleep)
                    break
                else:
                    # We're already inside the pre-margin window; proceed immediately.
                    break

            except Exception as e:
                logging.error(f"💥 Error parsing next epoch time: {e}")
                time.sleep(30)
                break  # re-fetch epoch info

        # If we broke because next epoch looked too far away, go back to top of outer loop.
        # Re-check that we actually *are* near the next epoch before running inference.
        info = fetch_last_epoch_info()
        if not isinstance(info, tuple) or len(info) != 6:
            logging.warning("⚠️ fetch_last_epoch_info() did not return 6 fields after reschedule; retrying in 45s.")
            time.sleep(45)
            continue

        (
            prev_epoch,
            prev_epoch_time,
            curr_epoch,
            curr_epoch_time,
            next_epoch,
            next_epoch_time,
        ) = info

        # If next_epoch_time is still far in the future, just loop again; don't run model
        try:
            clean_next = " ".join(str(next_epoch_time).split())
            next_dt = datetime.strptime(clean_next, "%m/%d/%Y %I:%M:%S %p")
            seconds_until = (next_dt - datetime.now()).total_seconds()
            if seconds_until > NEXT_TOO_FAR_SECONDS:
                logging.warning(
                    f"⚠️ After reschedule, next epoch still {seconds_until:.1f}s away. "
                    "Skipping inference this cycle."
                )
                time.sleep(60)
                continue
        except Exception as e:
            logging.error(f"💥 Error parsing next epoch time on reschedule: {e}")
            time.sleep(30)
            continue

        # Normal cycle: log header, build window, run model, log result
        _log_epoch_header(info)

        current_window = current_epoch_trend()
        result = infer_trend_for_next_epoch(current_window)

        if result is not None:
            # 3-class: 0=Bear, 1=Neutral, 2=Bull
            lbl = int(result.get("label", -1))
            if lbl == 0:
                label_str = "Bear"
            elif lbl == 2:
                label_str = "Bull"
            else:
                label_str = "Neutral"

            logging.info(
                f"Final Recorded Trend for Next Epoch: {next_epoch}  =  {label_str}  "
                f"(P_bear={result.get('P_bear', 0.0):.3f}, "
                f"P_neutral={result.get('P_neutral', 0.0):.3f}, "
                f"P_bull={result.get('P_bull', 0.0):.3f}, "
                f"engine={result.get('engine', 'baseline_?')})"
            )
        else:
            logging.info("⚠️ No trend result produced this cycle (skipped).")

        # Allow a little breathing room before we re-evaluate the schedule
        time.sleep(45)

