# main/engine/process/utils/DB_process_stats.py

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from main.engine.DB_round_fetch import fetch_last_epoch_info

# ---------------------------------------------------------------------
# Paths (match your existing pattern: SCRIPT_DIR / "../ts/json/...")
# main.engine.ts.json.round_record.json
# main.engine.ts.json.DB_rounds_trend.json
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE_DIR = SCRIPT_DIR.parent.parent

ROUND_RECORD_FILE = (ENGINE_DIR / "ts" / "json" / "round_record.json").resolve()
TRENDS_FILE = (ENGINE_DIR / "ts" / "json" / "DB_rounds_trend.json").resolve()
print(f"SCRIPT_DIR       = {SCRIPT_DIR}")
print(f"ENGINE_DIR       = {ENGINE_DIR}")
print(f"ROUND_RECORD_FILE= {ROUND_RECORD_FILE}")
print(f"TRENDS_FILE      = {TRENDS_FILE}")

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

log = logging.getLogger("DB_process_stats")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%I:%M:%S %p")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

LABELS = ("Bull", "Bear", "Neutral")

# Option 2 reset style: set this to the epoch you want stats to start from.
# When you want to reset coverage/accuracy, just change this number.
STATS_BASELINE_EPOCH = 324478  # e.g., 315231


def _parse_money(x: Any) -> Optional[float]:
    """Parses '$-115.22' / '115.22' / -115.22 -> float. Returns None if not parseable."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _truth_label_from_diff(price_diff: Optional[float]) -> str:
    """Ground truth label for an epoch based on its realized priceDifference."""
    if price_diff is None:
        return "Unknown"
    if price_diff > 0:
        return "Bull"
    if price_diff < 0:
        return "Bear"
    return "Neutral"


def _norm_label(x: Any) -> str:
    """Normalize prediction label variants to Bull/Bear/Neutral/Unknown."""
    s = str(x).strip().lower()
    if s in ("bull", "bullish", "up", "long", "1", "+1"):
        return "Bull"
    if s in ("bear", "bearish", "down", "short", "-1"):
        return "Bear"
    if s in ("neutral", "nuetral", "flat", "0", "none", "", "abstain"):
        return "Neutral"
    return "Unknown"


def _parse_next_epoch_time(x: Any) -> Optional[datetime]:
    """
    Accepts:
      - datetime
      - ISO strings (with optional Z)
      - your UI strings: "02/16/2026  07:20:56 PM"
    Returns datetime or None.
    """
    if x is None:
        return None
    if isinstance(x, datetime):
        return x

    s = str(x).strip()
    if not s or s == "N/A":
        return None

    # Try ISO first
    try:
        s_iso = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s_iso)
    except Exception:
        pass

    # Try your known format
    for fmt in ("%m/%d/%Y  %I:%M:%S %p", "%m/%d/%Y %I:%M:%S %p"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue

    return None


@dataclass
class StreakState:
    current_win: int = 0
    current_loss: int = 0
    max_win: int = 0
    max_loss: int = 0

    def reset(self) -> None:
        self.current_win = 0
        self.current_loss = 0

    def win(self) -> None:
        self.current_win += 1
        self.current_loss = 0
        self.max_win = max(self.max_win, self.current_win)

    def loss(self) -> None:
        self.current_loss += 1
        self.current_win = 0
        self.max_loss = max(self.max_loss, self.current_loss)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def epoch_stats() -> Dict[str, Any] | None:
    """
    Right/Wrong checker (Bull/Bear/Neutral):
    - Matches round_record.json.previousEpoch to DB_rounds_trend.json.next_epoch
    - Neutral predictions are abstains (⚪) and do NOT count toward accuracy
    - Reports:
        Directional coverage %
        Neutral rate %
        Bull precision %
        Bear precision %
        Confusion matrix
        Win/Loss streak tracking (directional only; Neutral breaks streak)
    """

    try:
        if not ROUND_RECORD_FILE.exists():
            log.error(f"❌ round_record.json not found: {ROUND_RECORD_FILE}")
            return None

        if not TRENDS_FILE.exists():
            log.error(f"❌ DB_rounds_trend.json not found: {TRENDS_FILE}")
            return None

        round_records = json.loads(ROUND_RECORD_FILE.read_text(encoding="utf-8"))
        trend_records = json.loads(TRENDS_FILE.read_text(encoding="utf-8"))

        # Build lookup: next_epoch → trend record
        forecast_lookup: Dict[int, dict] = {}
        for r in trend_records:
            try:
                ne = int(r.get("next_epoch"))
                forecast_lookup[ne] = r
            except Exception:
                continue

        # Confusion matrix: truth(row) x pred(col), including Neutral
        cm = {t: {p: 0 for p in LABELS} for t in LABELS}

        total_rows = 0
        directional_called = 0
        neutral_called = 0

        directional_correct = 0
        v2_correct = 0
        bull_pred = 0
        bear_pred = 0
        bull_tp = 0
        bear_tp = 0

        streak = StreakState()

        log.info("Epoch Stats Check (trend_method_v2_0):")
        log.info(f"Baseline: STATS_BASELINE_EPOCH={STATS_BASELINE_EPOCH} (epochs < baseline are ignored)")
        log.info("-" * 139)

        for rr in round_records:
            try:
                prev_epoch = int(rr.get("previousEpoch"))
            except Exception:
                continue

            if STATS_BASELINE_EPOCH and prev_epoch < STATS_BASELINE_EPOCH:
                continue

            forecast = forecast_lookup.get(prev_epoch)
            if not forecast:
                continue

            # Ground truth
            diff_val = _parse_money(rr.get("priceDifference"))
            truth = _truth_label_from_diff(diff_val)
            if truth not in LABELS:
                continue  # skip unknown truth rows

            # Prediction
            pred = _norm_label(forecast.get("trend", forecast.get("v2.0_trend", "Neutral")))
            if pred not in LABELS:
                pred = "Neutral"  # safest fallback

            total_rows += 1
            cm[truth][pred] += 1

            # Marking rules
            if pred == "Neutral":
                neutral_called += 1
                mark = "⚪"  # abstain
                streak.reset()  # abstain breaks streak
            else:
                directional_called += 1
                is_correct = (pred == truth)
                if is_correct:
                    directional_correct += 1
                    v2_correct += 1
                    mark = "✅"
                    streak.win()
                else:
                    mark = "❌"
                    streak.loss()

                if pred == "Bull":
                    bull_pred += 1
                    if truth == "Bull":
                        bull_tp += 1
                elif pred == "Bear":
                    bear_pred += 1
                    if truth == "Bear":
                        bear_tp += 1

            log.info(f"epoch {prev_epoch} | pred={pred:7s} | truth={truth:7s} | {mark}")

        log.info("-" * 139)

        # Metrics
        directional_accuracy = (directional_correct / directional_called) * 100 if directional_called else 0.0
        directional_coverage = (directional_called / total_rows) * 100 if total_rows else 0.0
        neutral_rate = (neutral_called / total_rows) * 100 if total_rows else 0.0

        bull_precision = (bull_tp / bull_pred) * 100 if bull_pred else 0.0
        bear_precision = (bear_tp / bear_pred) * 100 if bear_pred else 0.0

        log.info(
            f"Total Rows: {total_rows} | Directional Called: {directional_called} | Neutral Called: {neutral_called}"
        )
        log.info(
            f"Directional Accuracy: {directional_accuracy:.2f}% | "
            f"Directional Coverage: {directional_coverage:.2f}% | Neutral Rate: {neutral_rate:.2f}%"
        )
        log.info(f"Bull Precision: {bull_precision:.2f}% | Bear Precision: {bear_precision:.2f}%")

        # Confusion matrix
        log.info("Confusion Matrix (Truth x Pred):")
        header = f"{'':>8} | {'Bull':>6} {'Bear':>6} {'Neutral':>7}"
        log.info(header)
        log.info("-" * len(header))
        for t in LABELS:
            row = cm[t]
            log.info(f"{t:>8} | {row['Bull']:6d} {row['Bear']:6d} {row['Neutral']:7d}")

        # Streaks
        log.info(
            f"Win/Loss Streaks (directional only; Neutral breaks): "
            f"max_win={streak.max_win} | max_loss={streak.max_loss}"
        )

        return {
            "total_rows": total_rows,
            "directional_called": directional_called,
            "neutral_called": neutral_called,
            "directional_accuracy_pct": round(directional_accuracy, 2),
            "v2.0_trend": directional_called,
            "v2.0_correct": v2_correct,
            "directional_coverage_pct": round(directional_coverage, 2),
            "neutral_rate_pct": round(neutral_rate, 2),
            "bull_precision_pct": round(bull_precision, 2),
            "bear_precision_pct": round(bear_precision, 2),
            "confusion_matrix": cm,
            "streaks": {
                "max_win": streak.max_win,
                "max_loss": streak.max_loss,
            },
        }

    except Exception as e:
        log.error(f"💥 Exception in epoch_stats(): {e}")
        return None


def run_continuously() -> None:
    """
    Re-checks next epoch timing and runs epoch_stats once per epoch cycle.
    """
    while True:
        try:
            _, _, _, _, next_epoch, next_epoch_time = fetch_last_epoch_info()
            dt_next = _parse_next_epoch_time(next_epoch_time)

            if not next_epoch or dt_next is None:
                log.warning("⚠️ Next epoch info unavailable. Retrying in 30s.")
                time.sleep(30)
                continue

            # Sleep until 15 seconds before next epoch
            while True:
                _, _, _, _, next_epoch, next_epoch_time = fetch_last_epoch_info()
                dt_next = _parse_next_epoch_time(next_epoch_time)
                if dt_next is None:
                    time.sleep(10)
                    continue

                now = datetime.now(dt_next.tzinfo) if dt_next.tzinfo else datetime.now()
                seconds_until = (dt_next - now).total_seconds()

                target_sleep = max(0.0, seconds_until + 25.0)  # <-- true “25 seconds after epoch”

                # Your existing safety cap logic
                if target_sleep > 310:
                    log.info(
                        f"⏳ Sleep time ({target_sleep:.1f}s) exceeds 310 seconds. "
                        f"Sleeping 25 seconds then rechecking."
                    )
                    time.sleep(25)
                    continue

                if target_sleep > 0:
                    log.info(f"⏳ Sleeping {target_sleep:.1f}s until 25 seconds after next epoch.")
                    log.info("- - " * 25)
                    time.sleep(target_sleep)
                break

            log.info(f"⏭️ Next Epoch: {next_epoch}, Next Epoch Time: {dt_next}")
            epoch_stats()

            # Wait for the epoch to complete before next cycle re-sync
            time.sleep(45)

        except Exception as e:
            log.error(f"💥 Exception in run_continuously(): {e}")
            time.sleep(15)


if __name__ == "__main__":
    run_continuously()