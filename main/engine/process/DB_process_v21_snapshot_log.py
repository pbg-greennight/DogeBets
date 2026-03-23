# main/engine/process/DB_process_v21_snapshot_log.py

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


V21_LOG_DIR = Path(__file__).resolve().parent / "logs" / "v21"

V21_LIVE_EQ_TEXT = (
    "fast=s8 + 0.5*s83 | "
    "mid=1.5*s23 + 0.75*s53 | "
    "continuation=fast_score | "
    "reversal=fast_score - mid_score | "
    "branch=continuation if sign(s8)==sign(s83)!=0 else reversal_or_neutral | "
    "decision=continuation follows fast_score; contradiction branch uses reversal_score only when mid dominates"
)


def _to_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _time_prefix(decision_time: Any) -> str:
    dt = _parse_dt(decision_time)
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%I:%M:%S %p")


def _date_key(decision_time: Any) -> str:
    dt = _parse_dt(decision_time)
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d")


def _fmt_num(value: Any, decimals: int = 6) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


def _fmt_pairs(
    data: Mapping[str, Any],
    ordered_keys: list[str],
    decimals: int = 6,
) -> str:
    parts = []
    for key in ordered_keys:
        if key not in data:
            continue
        parts.append(f"{key}={_fmt_num(data[key], decimals)}")
    return " | ".join(parts)


def _fmt_any_pairs(
    data: Mapping[str, Any],
    ordered_keys: list[str],
) -> str:
    parts = []
    for key in ordered_keys:
        if key not in data:
            continue
        parts.append(f"{key}={data[key]}")
    return " | ".join(parts)


def build_v21_snapshot(
    *,
    source_epoch: int,
    predicted_next_epoch: int,
    decision_time: Any,
    next_epoch_time: Any,
    trend: str,
    confidence: float,
    model: str,
    notes: str,
    inputs: Mapping[str, Any],
    gauss: Mapping[str, Any],
    scores: Mapping[str, Any],
    mapped: Mapping[str, Any],
    hyst_meta: Mapping[str, Any],
    raw_keys: Mapping[str, Any],
    raw_slopes: Mapping[str, Any],
    eq_text: str = V21_LIVE_EQ_TEXT,
    schema_version: str = "v21_snapshot.v2",
    feature_flags: Optional[Mapping[str, Any]] = None,
    src_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    snapshot = {
        "schema_version": schema_version,
        "source_epoch": source_epoch,
        "predicted_next_epoch": predicted_next_epoch,
        "decision_time": _to_iso(decision_time),
        "next_epoch_time": _to_iso(next_epoch_time),
        "trend": trend,
        "confidence": confidence,
        "model": model,
        "notes": notes,
        "eq_text": eq_text,
        "inputs": dict(inputs),
        "gauss": dict(gauss),
        "scores": dict(scores),
        "mapped": dict(mapped),
        "hyst_meta": dict(hyst_meta),
        "raw_keys": dict(raw_keys),
        "raw_slopes": dict(raw_slopes),
        "feature_flags": dict(feature_flags or {}),
        "src_summary": dict(src_summary or {}),
    }
    return _json_ready(snapshot)


def format_v21_snapshot_text(snapshot: Mapping[str, Any]) -> str:
    prefix = _time_prefix(snapshot.get("decision_time"))

    def line(msg: str) -> str:
        return f"{prefix} - {msg}"

    inputs = snapshot.get("inputs", {}) or {}
    gauss = snapshot.get("gauss", {}) or {}
    scores = snapshot.get("scores", {}) or {}
    mapped = snapshot.get("mapped", {}) or {}
    hyst_meta = snapshot.get("hyst_meta", {}) or {}
    raw_keys = snapshot.get("raw_keys", {}) or {}
    raw_slopes = snapshot.get("raw_slopes", {}) or {}
    feature_flags = snapshot.get("feature_flags", {}) or {}
    src_summary = snapshot.get("src_summary", {}) or {}

    score_keys_new = [
        "fast_score",
        "mid_score",
        "continuation_score",
        "reversal_score",
        "decision_score",
        "confidence",
    ]
    score_keys_old = ["bull_score", "bear_score", "separation", "confidence"]

    eq_text = snapshot.get("eq_text")
    if not eq_text:
        eq_text = V21_LIVE_EQ_TEXT

    lines = [
        line(
            f"Epoch data from {snapshot.get('source_epoch')} --→ "
            f"Predict Next Epoch {snapshot.get('predicted_next_epoch')} | "
            f"trend={snapshot.get('trend')} | "
            f"confidence={_fmt_num(snapshot.get('confidence'), 3)} | "
            f"model={snapshot.get('model')} | "
            f"notes={snapshot.get('notes')}"
        ),
        line(
            f"[v21_live_ids] source_epoch={snapshot.get('source_epoch')} | "
            f"predicted_next_epoch={snapshot.get('predicted_next_epoch')} | "
            f"decision_time={snapshot.get('decision_time')} | "
            f"next_epoch_time={snapshot.get('next_epoch_time')} | "
            f"trend={snapshot.get('trend')} | "
            f"model={snapshot.get('model')} | "
            f"notes={snapshot.get('notes')}"
        ),
        line(f"[v21_live_eq] {eq_text}"),
        line(
            "[v21_live_inputs] "
            + _fmt_pairs(
                inputs,
                ["s8", "s23", "s53", "s83", "fast_min", "reversal_min", "mid_dom_ratio", "decision_min"],
                decimals=6,
            )
        ),
        line(
            "[v21_live_gauss] "
            + _fmt_pairs(
                gauss,
                ["g8", "g23", "g38", "g53", "g68", "g83"],
                decimals=4,
            )
        ),
    ]

    if any(key in scores for key in score_keys_new):
        lines.append(line("[v21_live_scores] " + _fmt_pairs(scores, score_keys_new, decimals=6)))
    elif scores:
        lines.append(line("[v21_live_scores] " + _fmt_pairs(scores, score_keys_old, decimals=6)))

    if mapped:
        lines.append(
            line(
                "[v21_live_mapped] "
                + _fmt_pairs(
                    mapped,
                    [
                        "compression",
                        "curvature_s23",
                        "curvature_s38",
                        "curvature_s53",
                        "curvature_s68",
                        "fan_alignment",
                        "fan_width",
                        "gcs_position",
                        "gcs_slope",
                        "gcs_width",
                        "hyst_stability",
                        "hyst_state",
                        "slope_s8",
                        "slope_s23",
                        "slope_s38",
                        "slope_s53",
                        "slope_s68",
                        "slope_s83",
                    ],
                    decimals=6,
                )
            )
        )

    if hyst_meta:
        lines.append(
            line(
                "[v21_live_hyst_meta] "
                + _fmt_any_pairs(
                    hyst_meta,
                    [
                        "align_tol_seconds",
                        "baseline_window_seconds",
                        "decision_time",
                        "missing_sigmas",
                        "skip_reason",
                        "skipped",
                        "tail_window_seconds",
                    ],
                )
            )
        )

    if raw_keys:
        lines.append(
            line(
                "[v21_live_raw_keys] "
                + " | ".join(f"{k}={raw_keys.get(k)}" for k in raw_keys.keys())
            )
        )

    if raw_slopes:
        lines.append(
            line(
                "[v21_live_raw_slopes] "
                + _fmt_pairs(raw_slopes, ["s8", "s23", "s38", "s53", "s68", "s83"], decimals=6)
            )
        )

    if feature_flags:
        lines.append(line("[v21_live_flags] " + " | ".join(f"{k}={v}" for k, v in feature_flags.items())))

    if src_summary:
        lines.append(
            line(
                "[v21_live_src] "
                + " | ".join(f"{k}={src_summary.get(k)}" for k in src_summary.keys())
            )
        )

    return "\n".join(lines) + "\n"


def write_v21_snapshot(
    snapshot: Mapping[str, Any],
    base_dir: Path | str = V21_LOG_DIR,
) -> tuple[Path, Path]:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    day = _date_key(snapshot.get("decision_time"))
    text_path = base_dir / f"v21_live_{day}.log"
    jsonl_path = base_dir / f"v21_live_{day}.jsonl"

    text_block = format_v21_snapshot_text(snapshot)

    with text_path.open("a", encoding="utf-8") as f:
        f.write(text_block)
        f.write("\n")

    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_ready(dict(snapshot)), ensure_ascii=False))
        f.write("\n")

    return text_path, jsonl_path
