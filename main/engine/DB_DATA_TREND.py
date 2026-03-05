# DogeBets\main\engine\DB_DATA_TREND.py
#
# Minimal “heartbeat” epoch logger:
# - run_continuously() stays synced to next epoch time using fetch_last_epoch_info()
# - sleeps until ~17 seconds before next epoch
# - epoch_trend() is the ONLY place that decides Bull/Bear/Neutral
#   (for now it always returns Neutral, by design for wiring tests)
# - save_forecast_log() persists the decision to DB_rounds_trend.json
#
# NOTE: We intentionally removed epoch_trend_calc(), rollback, and other unused code
# to match your stated goal: sync + save Neutral each epoch as a baseline.

import json
import time
import logging
from pathlib import Path
from datetime import datetime

from main.engine.DB_round_fetch import fetch_last_epoch_info

BASE_DIR = Path(__file__).resolve().parent
TS_DIR = (BASE_DIR / ".." / "ts").resolve()
TREND_LOG_FILE = TS_DIR / "json" / "DB_rounds_trend.json"

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%I:%M:%S %p", force=True)


def save_forecast_log(trend_label, confidence, next_epoch, model_version, mode):
    entry = {
        "timestamp": time.strftime("%m/%d/%Y %I:%M:%S %p"),
        "trend": trend_label,
        "confidence": round(confidence, 3),
        "next_epoch": next_epoch,
        "model_version": model_version,
        "mode": mode
    }

    history = []
    if TREND_LOG_FILE.exists():
        try:
            with open(TREND_LOG_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(entry)
    history = history[-100:]

    TREND_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TREND_LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)


def epoch_trend():
    """
    Standalone consumer (separate process-safe):
    Read the latest trend decision written by DB_process_orchestrator via _write_trend_out_json().

    Uses the SAME cfg() as orchestrator to locate TREND_OUT_JSON (no path drift).
    """
    try:
        _, _, _, _, next_epoch, _ = fetch_last_epoch_info()

        from main.engine.process.DB_process_trend import calculate_trend

        trend_label = "trend", "Neutral"
        confidence = "confidence", 0.0
        model_version = "model_id", ""
        mode = "reason", ""
        next_epoch_json = "next_epoch", None
        final_next_epoch = int(next_epoch_json) if next_epoch_json is not None else next_epoch

        logging.info(
            f"🔚 Final Recorded Trend for Next Epoch: {final_next_epoch}  =  {trend_label}   "
            f"| (confidence={confidence:.3f}, model={model_version}, notes={mode})"
        )
        logging.info("=" * 139)

        save_forecast_log(
            trend_label=trend_label,
            confidence=confidence,
            next_epoch=final_next_epoch,
            model_version=model_version,
            mode="from_TREND_OUT_JSON_cfg",
        )

        return trend_label, confidence, final_next_epoch

    except Exception as e:
        logging.exception(f"💥 Exception in epoch_trend(): {e}")
        return "Neutral", 0.0, None



def run_continuously():
    """
    Continuously:
      - refresh next epoch timing via fetch_last_epoch_info()
      - sleep until ~17s before next epoch
      - call epoch_trend() once per epoch

    Fix:
      - next_epoch_time now arrives as ISO-8601 string like '2026-02-02T15:55:17-05:00'
        (or sometimes as a datetime). We parse it robustly using fromisoformat().
      - We no longer overwrite year/month/day or add +1 day. The CSV already has the correct date.
    """
    while True:
        _, _, _, _, next_epoch, next_epoch_time = fetch_last_epoch_info()

        if next_epoch and next_epoch_time != "N/A":
            while True:
                try:
                    _, _, _, _, next_epoch, next_epoch_time = fetch_last_epoch_info()
                    if isinstance(next_epoch_time, datetime):
                        dt_next = next_epoch_time
                    else:
                        s = str(next_epoch_time).strip()
                        s = s.replace("Z", "+00:00")
                        dt_next = datetime.fromisoformat(s)

                    if dt_next.tzinfo is not None:
                        now = datetime.now(dt_next.tzinfo)
                    else:
                        now = datetime.now()

                    seconds_until = (dt_next - now).total_seconds()
                    target_sleep = max(0, seconds_until - 12)

                    if target_sleep > 310:
                        logging.info(
                            f"⏳ Sleep time ({target_sleep:.1f}s) exceeds 310 seconds. "
                            f"Sleeping 25 seconds then rechecking."
                        )
                        time.sleep(25)
                    elif target_sleep > 0:
                        logging.info(f"⏳ Sleeping {target_sleep:.1f}s until 15 seconds before next epoch.")
                        logging.info("- - " * 25)
                        time.sleep(target_sleep)
                        break
                    else:
                        break

                except Exception as e:
                    logging.error(f"💥 Error parsing next epoch time: {e}")
                    time.sleep(10)
                    break

            # ~12 seconds before next epoch: decide + save
            epoch_trend()


            # Wait for the epoch to complete before next cycle re-sync
            time.sleep(45)

        else:
            logging.warning("⚠️ Next epoch info unavailable. Retrying in 30s.")
            time.sleep(30)


if __name__ == "__main__":
    run_continuously()


