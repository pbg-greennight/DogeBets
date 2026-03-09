from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY_PATH = (SCRIPT_DIR / "models" / "model_predictions_history" / "model_predictions_history.jsonl").resolve()
DEFAULT_ROUND_PATH = (SCRIPT_DIR / "../ts/json/round_record.json").resolve()
DEFAULT_OUTPUT_DIR = (SCRIPT_DIR / "datasets").resolve()

COLLAPSE_COMPRESSION_THRESHOLD = -0.8
COLLAPSE_ALIGNMENT_THRESHOLD = 0.25
FLIP_SCORE_THRESHOLD = 0.05
WEAK_BEAR_COMPRESSION_THRESHOLD = 0.8

log = logging.getLogger("DB_process_dataset_export")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%I:%M:%S %p")


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
    "fan_to_slope_ratio",
    "collapse_flag",
    "weak_bull_exhaustion_flag",
    "weak_bear_exhaustion_flag",
]


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
    except ValueError:
        return None


def _norm_label(label: Any) -> str:
    val = str(label or "").strip().lower()
    if val in {"bull", "bullish", "up", "long", "1", "+1"}:
        return "Bull"
    if val in {"bear", "bearish", "down", "short", "-1"}:
        return "Bear"
    if val in {"neutral", "nuetral", "flat", "0", "none", "", "abstain"}:
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


def load_prediction_history(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL prediction history rows. Malformed lines are skipped safely."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        log.warning("⚠️ prediction history file not found: %s", path)
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                log.warning("⚠️ Skipping malformed JSONL line %d in %s", line_no, path)
                continue
            if isinstance(payload, dict):
                rows.append(payload)

    return rows


def load_round_record(path: Path) -> List[Dict[str, Any]]:
    """Load round records from JSON object/list. Malformed payloads return empty list."""
    if not path.exists():
        log.warning("⚠️ round record file not found: %s", path)
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("⚠️ Failed to parse round record JSON: %s", path)
        return []

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def _build_round_lookup(round_rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for row in round_rows:
        ne = _as_int(row.get("nextEpoch"))
        if ne is None:
            continue
        lookup[ne] = row
    return lookup


def _flatten_row(history_row: Dict[str, Any], model_row: Dict[str, Any], round_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    debug = model_row.get("debug") if isinstance(model_row.get("debug"), dict) else {}
    signals = debug.get("signals") if isinstance(debug.get("signals"), dict) else {}
    raw = model_row.get("raw_features_used") if isinstance(model_row.get("raw_features_used"), dict) else {}

    prediction = _norm_label(model_row.get("trend"))
    truth = "Unknown"
    start_price = None
    end_price = None
    price_difference = None
    round_current_timestamp = None
    round_next_epoch_time = None

    if round_row:
        price_difference = _as_float(round_row.get("priceDifference"))
        truth = _truth_from_price_diff(price_difference)
        start_price = _as_float(round_row.get("startPrice"))
        end_price = _as_float(round_row.get("endPrice"))
        round_current_timestamp = round_row.get("current_timestamp")
        round_next_epoch_time = round_row.get("nextEpochTime")

    directional_called = prediction in {"Bull", "Bear"}
    directional_correct = bool(directional_called and prediction == truth)
    correct = bool(prediction == truth and truth != "Unknown")

    bull_called = prediction == "Bull"
    bear_called = prediction == "Bear"
    neutral_called = prediction == "Neutral"

    fan_width = _as_float(signals.get("fan_width"))
    alignment = _as_float(signals.get("alignment"))
    compression_velocity = _as_float(signals.get("compression_velocity"))
    flip_score = _as_float(signals.get("flip_score"))
    slope_sum = _as_float(signals.get("slope_sum"))

    s23 = _as_float(raw.get("s23"))
    s53 = _as_float(raw.get("s53"))

    slope_sign_s23 = _sign(s23)
    slope_sign_s53 = _sign(s53)
    slope_disagreement = slope_sign_s23 != slope_sign_s53
    slope_gap = abs((s23 or 0.0) - (s53 or 0.0))
    abs_slope_sum = abs(slope_sum or 0.0)
    fan_to_slope_ratio = (fan_width or 0.0) / max(abs_slope_sum, 1e-9)

    collapse_flag = bool(
        (compression_velocity is not None and compression_velocity < COLLAPSE_COMPRESSION_THRESHOLD)
        and (alignment is not None and alignment < COLLAPSE_ALIGNMENT_THRESHOLD)
    )

    weak_bull_exhaustion_flag = bool(
        (slope_sum is not None and slope_sum > 0)
        and (compression_velocity is not None and compression_velocity < COLLAPSE_COMPRESSION_THRESHOLD)
        and (alignment is not None and alignment < COLLAPSE_ALIGNMENT_THRESHOLD)
        and (flip_score is not None and flip_score > FLIP_SCORE_THRESHOLD)
    )

    weak_bear_exhaustion_flag = bool(
        (slope_sum is not None and slope_sum < 0)
        and (compression_velocity is not None and compression_velocity > WEAK_BEAR_COMPRESSION_THRESHOLD)
        and (alignment is not None and alignment < COLLAPSE_ALIGNMENT_THRESHOLD)
        and (flip_score is not None and flip_score > FLIP_SCORE_THRESHOLD)
    )

    return {
        "model_id": str(model_row.get("model_id") or ""),
        "epoch": _as_int(history_row.get("epoch")),
        "next_epoch": _as_int(history_row.get("next_epoch")),
        "timestamp": history_row.get("timestamp"),
        "prediction": prediction,
        "confidence": _as_float(model_row.get("confidence")),
        "score": _as_float(model_row.get("score")),
        "reason": model_row.get("reason"),
        "truth": truth,
        "start_price": start_price,
        "end_price": end_price,
        "price_difference": price_difference,
        "round_current_timestamp": round_current_timestamp,
        "round_next_epoch_time": round_next_epoch_time,
        "correct": correct,
        "directional_called": directional_called,
        "directional_correct": directional_correct,
        "bull_called": bull_called,
        "bear_called": bear_called,
        "neutral_called": neutral_called,
        "bull_called_bear_truth": bool(bull_called and truth == "Bear"),
        "bear_called_bull_truth": bool(bear_called and truth == "Bull"),
        "neutral_called_bull_truth": bool(neutral_called and truth == "Bull"),
        "neutral_called_bear_truth": bool(neutral_called and truth == "Bear"),
        "bull_called_neutral_truth": bool(bull_called and truth == "Neutral"),
        "bear_called_neutral_truth": bool(bear_called and truth == "Neutral"),
        "regime": debug.get("regime"),
        "run_direction": debug.get("run_direction"),
        "fan_width": fan_width,
        "alignment": alignment,
        "compression_velocity": compression_velocity,
        "flip_score": flip_score,
        "slope_sum": slope_sum,
        "slope_magnitude": _as_float(signals.get("slope_magnitude")),
        "ratio_8_23_to_23_53": _as_float(signals.get("ratio_8_23_to_23_53")),
        "s23": s23,
        "s53": s53,
        "fan_width_raw": _as_float(raw.get("fan_width")),
        "alignment_raw": _as_float(raw.get("alignment")),
        "compression_velocity_raw": _as_float(raw.get("compression_velocity")),
        "flip_score_raw": _as_float(raw.get("flip_score")),
        "slope_sign_s23": slope_sign_s23,
        "slope_sign_s53": slope_sign_s53,
        "slope_disagreement": slope_disagreement,
        "slope_gap": slope_gap,
        "abs_slope_sum": abs_slope_sum,
        "fan_to_slope_ratio": fan_to_slope_ratio,
        "collapse_flag": collapse_flag,
        "weak_bull_exhaustion_flag": weak_bull_exhaustion_flag,
        "weak_bear_exhaustion_flag": weak_bear_exhaustion_flag,
    }


def build_gaussian_fan_dataset(
    history_path: Path,
    round_path: Path,
    output_dir: Path,
    model_id: str = "trend_method_v2_0",
) -> List[Dict[str, Any]]:
    """Build analysis rows for a selected model by joining prediction history with round truth."""
    history_rows = load_prediction_history(history_path)
    round_rows = load_round_record(round_path)
    round_lookup = _build_round_lookup(round_rows)

    out_rows: List[Dict[str, Any]] = []
    for history_row in history_rows:
        models = history_row.get("models")
        if not isinstance(models, list):
            continue

        matched_round = round_lookup.get(_as_int(history_row.get("next_epoch")) or -1)
        for model_row in models:
            if not isinstance(model_row, dict):
                continue
            current_model_id = str(model_row.get("model_id") or "")
            if current_model_id != model_id:
                continue
            out_rows.append(_flatten_row(history_row, model_row, matched_round))

    write_dataset_outputs(out_rows, output_dir)
    return out_rows


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _directional_accuracy_for_regime(rows: List[Dict[str, Any]], regime: str) -> float:
    regime_rows = [r for r in rows if str(r.get("regime") or "").upper() == regime]
    directional_rows = [r for r in regime_rows if r.get("directional_called") is True]
    if not directional_rows:
        return 0.0
    hits = sum(1 for r in directional_rows if r.get("directional_correct") is True)
    return _safe_ratio(hits, len(directional_rows))


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build summary metrics aligned with process stats style."""
    total_rows = len(rows)
    rows_with_joined_truth = sum(1 for r in rows if r.get("truth") in {"Bull", "Bear", "Neutral"})

    directional_called = sum(1 for r in rows if r.get("directional_called") is True)
    neutral_called = sum(1 for r in rows if r.get("neutral_called") is True)
    directional_correct = sum(1 for r in rows if r.get("directional_correct") is True)

    pred_bull = sum(1 for r in rows if r.get("prediction") == "Bull")
    pred_bear = sum(1 for r in rows if r.get("prediction") == "Bear")
    pred_neutral = sum(1 for r in rows if r.get("prediction") == "Neutral")

    correct_bull = sum(1 for r in rows if r.get("prediction") == "Bull" and r.get("truth") == "Bull")
    correct_bear = sum(1 for r in rows if r.get("prediction") == "Bear" and r.get("truth") == "Bear")

    truth_bull = sum(1 for r in rows if r.get("truth") == "Bull")
    truth_bear = sum(1 for r in rows if r.get("truth") == "Bear")
    truth_neutral = sum(1 for r in rows if r.get("truth") == "Neutral")

    summary = {
        "total_rows": total_rows,
        "rows_with_joined_truth": rows_with_joined_truth,
        "directional_called": directional_called,
        "neutral_called": neutral_called,
        "directional_accuracy": round(_safe_ratio(directional_correct, directional_called), 6),
        "directional_coverage": round(_safe_ratio(directional_called, total_rows), 6),
        "neutral_rate": round(_safe_ratio(neutral_called, total_rows), 6),
        "bull_precision": round(_safe_ratio(correct_bull, pred_bull), 6),
        "bear_precision": round(_safe_ratio(correct_bear, pred_bear), 6),
        "truth_bull": truth_bull,
        "truth_bear": truth_bear,
        "truth_neutral": truth_neutral,
        "pred_bull": pred_bull,
        "pred_bear": pred_bear,
        "pred_neutral": pred_neutral,
        "bull_called_bear_truth": sum(1 for r in rows if r.get("bull_called_bear_truth") is True),
        "bear_called_bull_truth": sum(1 for r in rows if r.get("bear_called_bull_truth") is True),
        "neutral_called_bull_truth": sum(1 for r in rows if r.get("neutral_called_bull_truth") is True),
        "neutral_called_bear_truth": sum(1 for r in rows if r.get("neutral_called_bear_truth") is True),
        "run_rows": sum(1 for r in rows if str(r.get("regime") or "").upper() == "RUN"),
        "reversal_rows": sum(1 for r in rows if str(r.get("regime") or "").upper() == "REVERSAL"),
        "noise_rows": sum(1 for r in rows if str(r.get("regime") or "").upper() == "NOISE"),
        "run_directional_accuracy": round(_directional_accuracy_for_regime(rows, "RUN"), 6),
        "reversal_directional_accuracy": round(_directional_accuracy_for_regime(rows, "REVERSAL"), 6),
        "noise_directional_accuracy": round(_directional_accuracy_for_regime(rows, "NOISE"), 6),
    }
    return summary


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_COLUMNS})


def write_dataset_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """Write full dataset, wrong directional rows, neutral misses, and summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "gaussian_fan_dataset.csv"
    wrong_path = output_dir / "gaussian_fan_wrong_predictions.csv"
    neutral_misses_path = output_dir / "gaussian_fan_neutral_misses.csv"
    summary_path = output_dir / "gaussian_fan_summary.json"

    wrong_rows = [r for r in rows if r.get("directional_called") is True and r.get("correct") is False]
    neutral_miss_rows = [
        r for r in rows if r.get("prediction") == "Neutral" and r.get("truth") in {"Bull", "Bear"}
    ]

    _write_csv(dataset_path, rows)
    _write_csv(wrong_path, wrong_rows)
    _write_csv(neutral_misses_path, neutral_miss_rows)

    summary = build_summary(rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return dataset_path, wrong_path, neutral_misses_path, summary_path


def _format_pct(value: float) -> str:
    return f"{(value or 0.0) * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Gaussian fan analysis datasets from prediction history + round truth")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH, help="Path to model prediction history JSONL")
    parser.add_argument("--round", dest="round_path", type=Path, default=DEFAULT_ROUND_PATH, help="Path to round_record JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Dataset output directory")
    parser.add_argument("--model-id", type=str, default="trend_method_v2_0", help="Model id to export")
    args = parser.parse_args()

    rows = build_gaussian_fan_dataset(
        history_path=args.history,
        round_path=args.round_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
    )
    summary = build_summary(rows)

    print(f"Total Rows: {summary['total_rows']}")
    print(f"Directional Called: {summary['directional_called']}")
    print(f"Neutral Called: {summary['neutral_called']}")
    print(f"Directional Accuracy: {_format_pct(summary['directional_accuracy'])}")
    print(f"Directional Coverage: {_format_pct(summary['directional_coverage'])}")
    print(f"Neutral Rate: {_format_pct(summary['neutral_rate'])}")
    print(f"Bull Precision: {_format_pct(summary['bull_precision'])}")
    print(f"Bear Precision: {_format_pct(summary['bear_precision'])}")
    print("Saved Files:")
    print(f"- {(args.output_dir / 'gaussian_fan_dataset.csv').resolve()}")
    print(f"- {(args.output_dir / 'gaussian_fan_wrong_predictions.csv').resolve()}")
    print(f"- {(args.output_dir / 'gaussian_fan_neutral_misses.csv').resolve()}")
    print(f"- {(args.output_dir / 'gaussian_fan_summary.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())