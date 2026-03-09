from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY_PATH = (SCRIPT_DIR / "models" / "model_predictions_history" / "model_predictions_history.jsonl").resolve()
DEFAULT_ROUND_PATH = (SCRIPT_DIR / "../ts/json/round_record.json").resolve()
DEFAULT_OUTPUT_DIR = (SCRIPT_DIR / "datasets").resolve()

# Tunable analysis constants.
COLLAPSE_COMPRESSION_THRESHOLD = -0.8
WEAK_ALIGNMENT_THRESHOLD = 0.25
FLIP_SCORE_ALERT_THRESHOLD = 0.05
RUN_ALIGNMENT_MIN = 0.50
NOISE_SLOPE_MAX = 0.002
FLIP_ZONE_RATIO_THRESHOLD = 0.0

MODEL_DEFAULT = "trend_method_v2_0"


CSV_COLUMNS = [
    "model_id",
    "epoch",
    "next_epoch",
    "timestamp",
    "prediction",
    "confidence",
    "score",
    "reason",
    "truth",
    "start_price",
    "end_price",
    "price_difference",
    "round_current_timestamp",
    "round_next_epoch_time",
    "correct",
    "directional_called",
    "directional_correct",
    "bull_called",
    "bear_called",
    "neutral_called",
    "bull_called_bear_truth",
    "bear_called_bull_truth",
    "neutral_called_bull_truth",
    "neutral_called_bear_truth",
    "bull_called_neutral_truth",
    "bear_called_neutral_truth",
    "regime",
    "run_direction",
    "fan_width",
    "alignment",
    "compression_velocity",
    "flip_score",
    "slope_sum",
    "slope_magnitude",
    "ratio_8_23_to_23_53",
    "s23",
    "s53",
    "fan_width_raw",
    "alignment_raw",
    "compression_velocity_raw",
    "flip_score_raw",
    "slope_sign_s23",
    "slope_sign_s53",
    "slope_disagreement",
    "slope_gap",
    "abs_slope_sum",
    "abs_compression_velocity",
    "abs_flip_score",
    "fan_to_slope_ratio",
    "collapse_flag",
    "weak_bull_exhaustion_flag",
    "weak_bear_exhaustion_flag",
    "possible_flip_zone",
    "possible_run_zone",
    "possible_transition_zone",
    "directional_error_type",
    "truth_direction_group",
    "prediction_direction_group",
]

log = logging.getLogger("DB_process_dataset_export")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%I:%M:%S %p")

def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("$", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _norm_direction(label: Any) -> str:
    txt = str(label or "").strip().lower()
    if txt in {"bull", "bullish", "up", "long", "+1", "1"}:
        return "Bull"
    if txt in {"bear", "bearish", "down", "short", "-1"}:
        return "Bear"
    if txt in {"neutral", "flat", "0", "none", "abstain", ""}:
        return "Neutral"
    return "Unknown"


def _truth_from_price_diff(price_diff: Optional[float]) -> str:
    if price_diff is None:
        return "Unknown"
    if price_diff > 0:
        return "Bull"
    if price_diff < 0:
        return "Bear"
    return "Neutral"


def _sign(value: Optional[float]) -> int:
    if value is None:
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0

def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _safe_mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [v for v in (_as_float(r.get(key)) for r in rows) if v is not None]
    return float(mean(vals)) if vals else 0.0


def _normalize_regime(value: Any) -> str:
    txt = str(value or "").strip().upper()
    if txt in {"RUN", "REVERSAL", "NOISE"}:
        return txt
    return "UNKNOWN"

def load_prediction_history(path: Path) -> List[Dict[str, Any]]:
    """Load prediction history JSONL lines safely."""
    if not path.exists():
        log.warning("Prediction history file not found: %s", path)
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                log.warning("Skipping malformed JSONL line %s in %s", line_no, path)
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)

    return rows


def load_round_record(path: Path) -> List[Dict[str, Any]]:
    """Load round record payload as list of row dicts."""
    if not path.exists():
        log.warning("Round record file not found: %s", path)
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Failed parsing round record JSON: %s", path)
        return []

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def flatten_model_rows(history_rows: Iterable[Dict[str, Any]], model_id: str) -> List[Dict[str, Any]]:
    """Flatten selected model rows from history payload."""
    flat: List[Dict[str, Any]] = []
    for history_row in history_rows:
        models = history_row.get("models")
        if not isinstance(models, list):
            continue

        for model_row in models:
            if not isinstance(model_row, dict):
                continue
            if str(model_row.get("model_id") or "") != model_id:
                continue
            flat.append(
                {
                    "history": history_row,
                    "model": model_row,
                }
            )
    return flat

def join_truth(
    flat_rows: Iterable[Dict[str, Any]],
    round_rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Join prediction rows with round-record truth data using:

        history.next_epoch  ->  round_record.nextEpoch

    Returns a list of flattened rows ready for derived feature processing.
    """

    # -------------------------------------------------
    # Build lookup table: nextEpoch -> round_row
    # -------------------------------------------------
    round_lookup: Dict[int, Dict[str, Any]] = {}

    for round_row in round_rows:
        if not isinstance(round_row, dict):
            continue

        next_epoch = _as_int(round_row.get("nextEpoch"))
        if next_epoch is None:
            continue

        round_lookup[next_epoch] = round_row

    output: List[Dict[str, Any]] = []

    # -------------------------------------------------
    # Iterate flattened prediction rows
    # -------------------------------------------------
    for block in flat_rows:

        if not isinstance(block, dict):
            continue

        history = block.get("history")
        model = block.get("model")

        if not isinstance(history, dict) or not isinstance(model, dict):
            continue

        debug = model.get("debug") if isinstance(model.get("debug"), dict) else {}
        signals = debug.get("signals") if isinstance(debug.get("signals"), dict) else {}
        raw = model.get("raw_features_used") if isinstance(model.get("raw_features_used"), dict) else {}

        next_epoch = _as_int(history.get("next_epoch"))

        matched_round: Optional[Dict[str, Any]] = None
        if next_epoch is not None:
            matched_round = round_lookup.get(next_epoch)

        # -------------------------------------------------
        # Extract round truth values
        # -------------------------------------------------
        price_difference = None
        start_price = None
        end_price = None
        round_ts = None
        round_next_epoch_time = None

        if matched_round:
            price_difference = _as_float(matched_round.get("priceDifference"))
            start_price = _as_float(matched_round.get("startPrice"))
            end_price = _as_float(matched_round.get("endPrice"))
            round_ts = matched_round.get("current_timestamp")
            round_next_epoch_time = matched_round.get("nextEpochTime")

        truth = _truth_from_price_diff(price_difference)
        prediction = _norm_direction(model.get("trend"))

        # -------------------------------------------------
        # Build flattened row
        # -------------------------------------------------
        row: Dict[str, Any] = {
            "model_id": str(model.get("model_id") or ""),
            "epoch": _as_int(history.get("epoch")),
            "next_epoch": next_epoch,
            "timestamp": history.get("timestamp"),

            "prediction": prediction,
            "confidence": _as_float(model.get("confidence")),
            "score": _as_float(model.get("score")),
            "reason": model.get("reason"),

            "truth": truth,
            "start_price": start_price,
            "end_price": end_price,
            "price_difference": price_difference,
            "round_current_timestamp": round_ts,
            "round_next_epoch_time": round_next_epoch_time,

            "regime": _normalize_regime(debug.get("regime")),
            "run_direction": str(debug.get("run_direction") or ""),

            # Signals
            "fan_width": _as_float(signals.get("fan_width")),
            "alignment": _as_float(signals.get("alignment")),
            "compression_velocity": _as_float(signals.get("compression_velocity")),
            "flip_score": _as_float(signals.get("flip_score")),
            "slope_sum": _as_float(signals.get("slope_sum")),
            "slope_magnitude": _as_float(signals.get("slope_magnitude")),
            "ratio_8_23_to_23_53": _as_float(signals.get("ratio_8_23_to_23_53")),

            # Raw features
            "s23": _as_float(raw.get("s23")),
            "s53": _as_float(raw.get("s53")),
            "fan_width_raw": _as_float(raw.get("fan_width")),
            "alignment_raw": _as_float(raw.get("alignment")),
            "compression_velocity_raw": _as_float(raw.get("compression_velocity")),
            "flip_score_raw": _as_float(raw.get("flip_score")),
        }

        # -------------------------------------------------
        # Evaluation flags
        # -------------------------------------------------
        row["directional_called"] = row["prediction"] in {"Bull", "Bear"}
        row["directional_correct"] = bool(
            row["directional_called"] and row["prediction"] == row["truth"]
        )

        row["correct"] = bool(
            row["prediction"] == row["truth"] and row["truth"] != "Unknown"
        )

        row["bull_called"] = row["prediction"] == "Bull"
        row["bear_called"] = row["prediction"] == "Bear"
        row["neutral_called"] = row["prediction"] == "Neutral"

        # -------------------------------------------------
        # Error taxonomy flags
        # -------------------------------------------------
        row["bull_called_bear_truth"] = bool(row["bull_called"] and row["truth"] == "Bear")
        row["bear_called_bull_truth"] = bool(row["bear_called"] and row["truth"] == "Bull")
        row["neutral_called_bull_truth"] = bool(row["neutral_called"] and row["truth"] == "Bull")
        row["neutral_called_bear_truth"] = bool(row["neutral_called"] and row["truth"] == "Bear")
        row["bull_called_neutral_truth"] = bool(row["bull_called"] and row["truth"] == "Neutral")
        row["bear_called_neutral_truth"] = bool(row["bear_called"] and row["truth"] == "Neutral")

        output.append(row)

    return output


def add_derived_columns(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append derived diagnostics columns used by v2.x/v3.0 analysis."""
    derived_rows: List[Dict[str, Any]] = []
    for row in rows:
        s23 = _as_float(row.get("s23"))
        s53 = _as_float(row.get("s53"))
        slope_sum = _as_float(row.get("slope_sum")) or 0.0
        fan_width = _as_float(row.get("fan_width")) or 0.0
        alignment = _as_float(row.get("alignment"))
        compression_velocity = _as_float(row.get("compression_velocity"))
        flip_score = _as_float(row.get("flip_score"))

        slope_sign_s23 = _sign(s23)
        slope_sign_s53 = _sign(s53)
        slope_disagreement = slope_sign_s23 != slope_sign_s53
        slope_gap = abs((s23 or 0.0) - (s53 or 0.0))
        abs_slope_sum = abs(slope_sum)
        abs_compression_velocity = abs(compression_velocity or 0.0)
        abs_flip_score = abs(flip_score or 0.0)
        fan_to_slope_ratio = fan_width / max(abs_slope_sum, 1e-9)

        collapse_flag = bool(
            compression_velocity is not None
            and alignment is not None
            and compression_velocity < COLLAPSE_COMPRESSION_THRESHOLD
            and alignment < WEAK_ALIGNMENT_THRESHOLD
        )
        weak_bull_exhaustion_flag = bool(
            slope_sum > 0
            and compression_velocity is not None
            and alignment is not None
            and flip_score is not None
            and compression_velocity < COLLAPSE_COMPRESSION_THRESHOLD
            and alignment < WEAK_ALIGNMENT_THRESHOLD
            and flip_score > FLIP_SCORE_ALERT_THRESHOLD
        )
        weak_bear_exhaustion_flag = bool(
            slope_sum < 0
            and compression_velocity is not None
            and alignment is not None
            and flip_score is not None
            and compression_velocity > abs(COLLAPSE_COMPRESSION_THRESHOLD)
            and alignment < WEAK_ALIGNMENT_THRESHOLD
            and flip_score > FLIP_SCORE_ALERT_THRESHOLD
        )
        possible_flip_zone = bool(
            slope_disagreement
            or (flip_score is not None and flip_score > FLIP_SCORE_ALERT_THRESHOLD)
            or (alignment is not None and alignment < WEAK_ALIGNMENT_THRESHOLD)
            or (fan_to_slope_ratio < FLIP_ZONE_RATIO_THRESHOLD)
        )
        possible_run_zone = bool(
            abs_slope_sum > NOISE_SLOPE_MAX
            and alignment is not None
            and alignment >= RUN_ALIGNMENT_MIN
            and not slope_disagreement
        )
        possible_transition_zone = bool((not possible_run_zone) and possible_flip_zone)

        truth = row.get("truth") if row.get("truth") in {"Bull", "Bear", "Neutral"} else "Unknown"
        pred = row.get("prediction") if row.get("prediction") in {"Bull", "Bear", "Neutral"} else "Unknown"

        directional_error_type = "none"
        if pred == "Bull" and truth == "Bear":
            directional_error_type = "bull_called_bear_truth"
        elif pred == "Bear" and truth == "Bull":
            directional_error_type = "bear_called_bull_truth"
        elif pred == "Neutral" and truth == "Bull":
            directional_error_type = "neutral_called_bull_truth"
        elif pred == "Neutral" and truth == "Bear":
            directional_error_type = "neutral_called_bear_truth"
        elif pred == "Bull" and truth == "Bull":
            directional_error_type = "correct_bull"
        elif pred == "Bear" and truth == "Bear":
            directional_error_type = "correct_bear"
        elif pred == "Neutral" and truth == "Neutral":
            directional_error_type = "correct_neutral"

        row.update(
            {
                "slope_sign_s23": slope_sign_s23,
                "slope_sign_s53": slope_sign_s53,
                "slope_disagreement": slope_disagreement,
                "slope_gap": slope_gap,
                "abs_slope_sum": abs_slope_sum,
                "abs_compression_velocity": abs_compression_velocity,
                "abs_flip_score": abs_flip_score,
                "fan_to_slope_ratio": fan_to_slope_ratio,
                "collapse_flag": collapse_flag,
                "weak_bull_exhaustion_flag": weak_bull_exhaustion_flag,
                "weak_bear_exhaustion_flag": weak_bear_exhaustion_flag,
                "possible_flip_zone": possible_flip_zone,
                "possible_run_zone": possible_run_zone,
                "possible_transition_zone": possible_transition_zone,
                "truth_direction_group": truth,
                "prediction_direction_group": pred,
                "directional_error_type": directional_error_type,
            }
        )
        derived_rows.append(row)
    return derived_rows


def _build_group_stats(rows: Sequence[Dict[str, Any]], group_name: str, group_value: str) -> Dict[str, Any]:
    directional_rows = [r for r in rows if r.get("directional_called")]
    pred_bull = [r for r in rows if r.get("prediction") == "Bull"]
    pred_bear = [r for r in rows if r.get("prediction") == "Bear"]
    pred_neutral = [r for r in rows if r.get("prediction") == "Neutral"]

    correct_bull = [r for r in rows if r.get("prediction") == "Bull" and r.get("truth") == "Bull"]
    correct_bear = [r for r in rows if r.get("prediction") == "Bear" and r.get("truth") == "Bear"]

    return {
        "group_name": group_name,
        "group_value": group_value,
        "row_count": len(rows),
        "directional_called": len(directional_rows),
        "directional_correct": sum(1 for r in directional_rows if r.get("directional_correct")),
        "directional_accuracy": _safe_ratio(
            sum(1 for r in directional_rows if r.get("directional_correct")),
            len(directional_rows),
        ),
        "coverage_within_export_set": 0.0,
        "bull_precision": _safe_ratio(len(correct_bull), len(pred_bull)),
        "bear_precision": _safe_ratio(len(correct_bear), len(pred_bear)),
        "neutral_rate": _safe_ratio(len(pred_neutral), len(rows)),
        "mean_s23": _safe_mean(rows, "s23"),
        "mean_s53": _safe_mean(rows, "s53"),
        "mean_slope_sum": _safe_mean(rows, "slope_sum"),
        "mean_fan_width": _safe_mean(rows, "fan_width"),
        "mean_alignment": _safe_mean(rows, "alignment"),
        "mean_compression_velocity": _safe_mean(rows, "compression_velocity"),
        "mean_flip_score": _safe_mean(rows, "flip_score"),
        "mean_ratio_8_23_to_23_53": _safe_mean(rows, "ratio_8_23_to_23_53"),
        "mean_slope_gap": _safe_mean(rows, "slope_gap"),
        "collapse_flag_rate": _safe_ratio(sum(1 for r in rows if r.get("collapse_flag")), len(rows)),
        "weak_bull_exhaustion_flag_rate": _safe_ratio(
            sum(1 for r in rows if r.get("weak_bull_exhaustion_flag")), len(rows)
        ),
        "weak_bear_exhaustion_flag_rate": _safe_ratio(
            sum(1 for r in rows if r.get("weak_bear_exhaustion_flag")), len(rows)
        ),
    }

def _group_rows(rows: Sequence[Dict[str, Any]], key_name: str, values: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: List[Dict[str, Any]] = []
    for val in values:
        subset = [r for r in rows if str(r.get(key_name)) == val]
        grouped.append(_build_group_stats(subset, key_name, val))
    total = max(1, len(rows))
    for record in grouped:
        record["coverage_within_export_set"] = _safe_ratio(record["row_count"], total)
    return grouped

def build_regime_breakdown(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _group_rows(rows, "regime", ["RUN", "REVERSAL", "NOISE", "UNKNOWN"])

def build_error_breakdown(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    error_values = [
        "bull_called_bear_truth",
        "bear_called_bull_truth",
        "neutral_called_bull_truth",
        "neutral_called_bear_truth",
        "correct_bull",
        "correct_bear",
        "correct_neutral",
    ]
    out = _group_rows(rows, "directional_error_type", error_values)

    wrong_bull_bear = [r for r in rows if r.get("directional_error_type") == "bull_called_bear_truth"]
    correct_bull = [r for r in rows if r.get("directional_error_type") == "correct_bull"]
    for label, subset in [
        ("v2_1_wrong_bull_to_bear", wrong_bull_bear),
        ("v2_1_correct_bull", correct_bull),
    ]:
        stats = _build_group_stats(subset, "v2_1_diagnostic_group", label)
        stats["slope_disagreement_rate"] = _safe_ratio(
            sum(1 for r in subset if r.get("slope_disagreement")),
            len(subset),
        )
        out.append(stats)
    return out

def build_run_flip_breakdown(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("possible_run_zone", [r for r in rows if r.get("possible_run_zone")]),
        ("possible_flip_zone", [r for r in rows if r.get("possible_flip_zone")]),
        ("possible_transition_zone", [r for r in rows if r.get("possible_transition_zone")]),
    ]
    output = []
    total = max(1, len(rows))
    for name, subset in groups:
        stats = _build_group_stats(subset, "proxy_zone", name)
        stats["coverage_within_export_set"] = _safe_ratio(stats["row_count"], total)
        output.append(stats)
    return output

def _max_streak(rows: Sequence[Dict[str, Any]], target: str) -> int:
    streak = 0
    best = 0
    for row in rows:
        if not row.get("directional_called"):
            streak = 0
            continue
        if target == "win" and row.get("directional_correct"):
            streak += 1
        elif target == "loss" and not row.get("directional_correct"):
            streak += 1
        else:
            streak = 0
        best = max(best, streak)
    return best

def build_main_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_rows = len(rows)
    joined_truth_rows = sum(1 for r in rows if r.get("truth") in {"Bull", "Bear", "Neutral"})
    directional_called = sum(1 for r in rows if r.get("directional_called"))
    neutral_called = sum(1 for r in rows if r.get("neutral_called"))
    directional_correct = sum(1 for r in rows if r.get("directional_correct"))

    bull_pred_bull_truth = sum(1 for r in rows if r.get("prediction") == "Bull" and r.get("truth") == "Bull")
    bull_pred_bear_truth = sum(1 for r in rows if r.get("prediction") == "Bull" and r.get("truth") == "Bear")
    bear_pred_bull_truth = sum(1 for r in rows if r.get("prediction") == "Bear" and r.get("truth") == "Bull")
    bear_pred_bear_truth = sum(1 for r in rows if r.get("prediction") == "Bear" and r.get("truth") == "Bear")
    neutral_pred_bull_truth = sum(1 for r in rows if r.get("prediction") == "Neutral" and r.get("truth") == "Bull")
    neutral_pred_bear_truth = sum(1 for r in rows if r.get("prediction") == "Neutral" and r.get("truth") == "Bear")

    pred_bull_count = sum(1 for r in rows if r.get("prediction") == "Bull")
    pred_bear_count = sum(1 for r in rows if r.get("prediction") == "Bear")
    pred_neutral_count = sum(1 for r in rows if r.get("prediction") == "Neutral")

    truth_bull_count = sum(1 for r in rows if r.get("truth") == "Bull")
    truth_bear_count = sum(1 for r in rows if r.get("truth") == "Bear")
    truth_neutral_count = sum(1 for r in rows if r.get("truth") == "Neutral")

    run_rows = [r for r in rows if r.get("regime") == "RUN"]
    reversal_rows = [r for r in rows if r.get("regime") == "REVERSAL"]
    noise_rows = [r for r in rows if r.get("regime") == "NOISE"]

    return {
        "total_rows": total_rows,
        "joined_truth_rows": joined_truth_rows,
        "directional_called": directional_called,
        "neutral_called": neutral_called,
        "directional_accuracy": _safe_ratio(directional_correct, directional_called),
        "directional_coverage": _safe_ratio(directional_called, total_rows),
        "neutral_rate": _safe_ratio(neutral_called, total_rows),
        "bull_precision": _safe_ratio(bull_pred_bull_truth, pred_bull_count),
        "bear_precision": _safe_ratio(bear_pred_bear_truth, pred_bear_count),
        "truth_bull_count": truth_bull_count,
        "truth_bear_count": truth_bear_count,
        "truth_neutral_count": truth_neutral_count,
        "pred_bull_count": pred_bull_count,
        "pred_bear_count": pred_bear_count,
        "pred_neutral_count": pred_neutral_count,
        "bull_pred_bull_truth": bull_pred_bull_truth,
        "bull_pred_bear_truth": bull_pred_bear_truth,
        "bear_pred_bull_truth": bear_pred_bull_truth,
        "bear_pred_bear_truth": bear_pred_bear_truth,
        "neutral_pred_bull_truth": neutral_pred_bull_truth,
        "neutral_pred_bear_truth": neutral_pred_bear_truth,
        "run_rows": len(run_rows),
        "reversal_rows": len(reversal_rows),
        "noise_rows": len(noise_rows),
        "run_directional_accuracy": _safe_ratio(
            sum(1 for r in run_rows if r.get("directional_correct")),
            sum(1 for r in run_rows if r.get("directional_called")),
        ),
        "reversal_directional_accuracy": _safe_ratio(
            sum(1 for r in reversal_rows if r.get("directional_correct")),
            sum(1 for r in reversal_rows if r.get("directional_called")),
        ),
        "noise_directional_accuracy": _safe_ratio(
            sum(1 for r in noise_rows if r.get("directional_correct")),
            sum(1 for r in noise_rows if r.get("directional_called")),
        ),
        "possible_run_zone_rows": sum(1 for r in rows if r.get("possible_run_zone")),
        "possible_flip_zone_rows": sum(1 for r in rows if r.get("possible_flip_zone")),
        "possible_transition_zone_rows": sum(1 for r in rows if r.get("possible_transition_zone")),
        "consecutive_directional_wins_max": _max_streak(rows, "win"),
        "consecutive_directional_losses_max": _max_streak(rows, "loss"),
        "neutral_breaks_streak": True,
    }


def run_threshold_scan(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analysis-only threshold scan for Bull overrides."""
    compression_levels = [-0.4, -0.6, -0.8, -1.0, -1.2]
    alignment_levels = [0.15, 0.20, 0.25, 0.30, 0.40]
    flip_levels = [0.03, 0.05, 0.07, 0.10]

    results: List[Dict[str, Any]] = []
    combos = itertools.product(compression_levels, alignment_levels, flip_levels)

    for c_thr, a_thr, f_thr in combos:
        candidates = [
            r
            for r in rows
            if r.get("prediction") == "Bull"
            and (_as_float(r.get("compression_velocity")) or 0.0) <= c_thr
            and (_as_float(r.get("alignment")) or 0.0) <= a_thr
            and (_as_float(r.get("flip_score")) or 0.0) >= f_thr
        ]
        for variant in ["bull_to_neutral", "bull_to_bear"]:
            simulated = []
            candidate_keys = {(r.get("epoch"), r.get("next_epoch"), r.get("timestamp")) for r in candidates}
            for row in rows:
                clone = dict(row)
                key = (row.get("epoch"), row.get("next_epoch"), row.get("timestamp"))
                if key in candidate_keys and row.get("prediction") == "Bull":
                    clone["prediction"] = "Neutral" if variant == "bull_to_neutral" else "Bear"
                    clone["directional_called"] = clone["prediction"] in {"Bull", "Bear"}
                    clone["directional_correct"] = bool(
                        clone["directional_called"] and clone["prediction"] == clone.get("truth")
                    )
                simulated.append(clone)

            directional_called = sum(1 for r in simulated if r.get("directional_called"))
            directional_correct = sum(1 for r in simulated if r.get("directional_correct"))
            bull_pred = sum(1 for r in simulated if r.get("prediction") == "Bull")
            bear_pred = sum(1 for r in simulated if r.get("prediction") == "Bear")
            bull_correct = sum(1 for r in simulated if r.get("prediction") == "Bull" and r.get("truth") == "Bull")
            bear_correct = sum(1 for r in simulated if r.get("prediction") == "Bear" and r.get("truth") == "Bear")

            results.append(
                {
                    "variant": variant,
                    "compression_threshold": c_thr,
                    "alignment_threshold": a_thr,
                    "flip_score_threshold": f_thr,
                    "changed_rows_count": len(candidates),
                    "new_directional_called": directional_called,
                    "new_directional_accuracy": _safe_ratio(directional_correct, directional_called),
                    "new_bull_precision": _safe_ratio(bull_correct, bull_pred),
                    "new_bear_precision": _safe_ratio(bear_correct, bear_pred),
                }
            )

    return results


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    regime_breakdown: Sequence[Dict[str, Any]],
    error_breakdown: Sequence[Dict[str, Any]],
    run_flip_breakdown: Sequence[Dict[str, Any]],
    threshold_scan: Sequence[Dict[str, Any]],
) -> Dict[str, Path]:
    """Write all required datasets and summary files."""
    paths = {
        "dataset": output_dir / "gaussian_fan_dataset.csv",
        "wrong": output_dir / "gaussian_fan_wrong_predictions.csv",
        "neutral_misses": output_dir / "gaussian_fan_neutral_misses.csv",
        "summary": output_dir / "gaussian_fan_summary.json",
        "regime_breakdown": output_dir / "gaussian_fan_regime_breakdown.csv",
        "error_breakdown": output_dir / "gaussian_fan_error_breakdown.csv",
        "run_flip_breakdown": output_dir / "gaussian_fan_run_flip_breakdown.csv",
        "threshold_scan": output_dir / "gaussian_fan_threshold_scan.csv",
    }


    wrong_rows = [r for r in rows if r.get("directional_called") and not r.get("directional_correct")]
    neutral_misses = [r for r in rows if r.get("prediction") == "Neutral" and r.get("truth") in {"Bull", "Bear"}]

    _write_csv(paths["dataset"], rows, CSV_COLUMNS)
    _write_csv(paths["wrong"], wrong_rows, CSV_COLUMNS)
    _write_csv(paths["neutral_misses"], neutral_misses, CSV_COLUMNS)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    group_fields = list(regime_breakdown[0].keys()) if regime_breakdown else ["group_name", "group_value", "row_count"]
    _write_csv(paths["regime_breakdown"], regime_breakdown, group_fields)
    _write_csv(paths["error_breakdown"], error_breakdown, list(error_breakdown[0].keys()) if error_breakdown else group_fields)
    _write_csv(paths["run_flip_breakdown"], run_flip_breakdown, list(run_flip_breakdown[0].keys()) if run_flip_breakdown else group_fields)
    _write_csv(paths["threshold_scan"], threshold_scan, list(threshold_scan[0].keys()) if threshold_scan else ["variant"])

    return paths


def _pct(value: float) -> str:
    return f"{(value or 0.0) * 100:.2f}%"

def print_terminal_summary(
    summary: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    error_breakdown: Sequence[Dict[str, Any]],
    paths: Dict[str, Path],
) -> None:
    """Print readable diagnostics sections in terminal."""
    print("\n=== Gaussian Fan Export: Headline Stats ===")
    print(f"Total Rows: {summary['total_rows']}")
    print(f"Directional Called: {summary['directional_called']}")
    print(f"Neutral Called: {summary['neutral_called']}")
    print(f"Directional Accuracy: {_pct(summary['directional_accuracy'])}")
    print(f"Directional Coverage: {_pct(summary['directional_coverage'])}")
    print(f"Neutral Rate: {_pct(summary['neutral_rate'])}")
    print(f"Bull Precision: {_pct(summary['bull_precision'])}")
    print(f"Bear Precision: {_pct(summary['bear_precision'])}")

    print("\n=== Confusion-style Counts ===")
    print(f"Bull predicted / Bull truth: {summary['bull_pred_bull_truth']}")
    print(f"Bull predicted / Bear truth: {summary['bull_pred_bear_truth']}")
    print(f"Bear predicted / Bull truth: {summary['bear_pred_bull_truth']}")
    print(f"Bear predicted / Bear truth: {summary['bear_pred_bear_truth']}")
    print(f"Neutral predicted / Bull truth: {summary['neutral_pred_bull_truth']}")
    print(f"Neutral predicted / Bear truth: {summary['neutral_pred_bear_truth']}")

    print("\n=== Regime Breakdown Summary ===")
    print(f"RUN rows / accuracy: {summary['run_rows']} / {_pct(summary['run_directional_accuracy'])}")
    print(f"REVERSAL rows / accuracy: {summary['reversal_rows']} / {_pct(summary['reversal_directional_accuracy'])}")
    print(f"NOISE rows / accuracy: {summary['noise_rows']} / {_pct(summary['noise_directional_accuracy'])}")

    wrong_bull = next((r for r in error_breakdown if r.get("group_value") == "v2_1_wrong_bull_to_bear"), None)
    correct_bull = next((r for r in error_breakdown if r.get("group_value") == "v2_1_correct_bull"), None)
    print("\n=== V2_1 Bear Improvement Diagnostic ===")
    print(f"wrong Bull->Bear rows: {int((wrong_bull or {}).get('row_count', 0))}")
    print(f"correct Bull rows: {int((correct_bull or {}).get('row_count', 0))}")
    if wrong_bull and correct_bull:
        print(
            "mean compression_velocity (wrong vs correct): "
            f"{wrong_bull.get('mean_compression_velocity', 0.0):.5f} vs {correct_bull.get('mean_compression_velocity', 0.0):.5f}"
        )
        print(
            "mean alignment (wrong vs correct): "
            f"{wrong_bull.get('mean_alignment', 0.0):.5f} vs {correct_bull.get('mean_alignment', 0.0):.5f}"
        )
        print(
            "mean flip_score (wrong vs correct): "
            f"{wrong_bull.get('mean_flip_score', 0.0):.5f} vs {correct_bull.get('mean_flip_score', 0.0):.5f}"
        )
        print(
            "weak_bull_exhaustion_flag rate (wrong vs correct): "
            f"{_pct(wrong_bull.get('weak_bull_exhaustion_flag_rate', 0.0))} vs "
            f"{_pct(correct_bull.get('weak_bull_exhaustion_flag_rate', 0.0))}"
        )

    print("\n=== Saved Files ===")
    for key in [
        "dataset",
        "wrong",
        "neutral_misses",
        "summary",
        "regime_breakdown",
        "error_breakdown",
        "run_flip_breakdown",
        "threshold_scan",
    ]:
        print(f"- {paths[key].resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gaussian fan dataset exporter + diagnostics")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH, help="Prediction history JSONL path")
    parser.add_argument("--round", dest="round_path", type=Path, default=DEFAULT_ROUND_PATH, help="Round record JSON path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model-id", type=str, default=MODEL_DEFAULT, help="Model id to export")
    args = parser.parse_args()

    history_rows = load_prediction_history(args.history)
    round_rows = load_round_record(args.round_path)
    flat_rows = flatten_model_rows(history_rows, args.model_id)
    joined = join_truth(flat_rows, round_rows)
    rows = add_derived_columns(joined)

    summary = build_main_summary(rows)
    regime_breakdown = build_regime_breakdown(rows)
    error_breakdown = build_error_breakdown(rows)
    run_flip_breakdown = build_run_flip_breakdown(rows)
    threshold_scan = run_threshold_scan(rows)

    paths = write_outputs(
        output_dir=args.output_dir,
        rows=rows,
        summary=summary,
        regime_breakdown=regime_breakdown,
        error_breakdown=error_breakdown,
        run_flip_breakdown=run_flip_breakdown,
        threshold_scan=threshold_scan,
    )
    summary = build_main_summary(rows)
    print_terminal_summary(summary, rows, error_breakdown, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())