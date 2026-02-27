"""main/engine/process/DB_process_csv.py

CSV schema + writers for DB_DATA_PROCESS refactor.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from DB_process_metrics import _to_float_or_none, snapshot_metrics
from DB_process_slicing import slice_tail_window_with_fallback
from DB_process_types import EpochTiming, TrendDecision, Windows


def _ensure_output_dir(config: Dict[str, Any]) -> Path:
    out_dir: Path = config["OUTPUT_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _wide_headers(registry: List[Dict[str, Any]]) -> List[str]:
    cols = [
        "decision_ts_est",
        "next_epoch_time_est",
        "curr_epoch",
        "next_epoch",
        "full_start_est",
        "full_end_est",
    ]
    for r in registry:
        s = r["sigma"]
        cols += [
            f"g{s}_tail60_last", f"g{s}_tail60_delta", f"g{s}_tail60_slope", f"g{s}_tail60_curve", f"g{s}_tail60_tag",
            f"g{s}_tail30_last", f"g{s}_tail30_delta", f"g{s}_tail30_slope", f"g{s}_tail30_curve", f"g{s}_tail30_tag",
            f"g{s}_tail20_last", f"g{s}_tail20_delta", f"g{s}_tail20_slope", f"g{s}_tail20_curve", f"g{s}_tail20_tag",
            f"g{s}_tail10_last", f"g{s}_tail10_delta", f"g{s}_tail10_slope", f"g{s}_tail10_curve", f"g{s}_tail10_tag",
            f"g{s}_tail0_last",  f"g{s}_tail0_delta",  f"g{s}_tail0_slope",  f"g{s}_tail0_curve",  f"g{s}_tail0_tag",
        ]
    cols += ["trend", "confidence", "trend_model", "trend_notes"]
    return cols


def write_wide_snapshot(
    config: Dict[str, Any],
    timing: EpochTiming,
    windows: Windows,
    decision_dt: datetime,
    registry: List[Dict[str, Any]],
    per_sigma_all: Dict[int, Dict[str, Any]],
    tail_seconds_list: List[int],
    trend: TrendDecision,
) -> None:
    out_dir = _ensure_output_dir(config)
    path = out_dir / config["WIDE_CSV"]

    file_exists = path.exists()

    tail_metrics: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for tail_sec in tail_seconds_list:
        start_dt = decision_dt - timedelta(seconds=int(tail_sec))
        end_dt = decision_dt
        for sigma in sorted(per_sigma_all.keys()):
            ts_all = per_sigma_all[sigma]["ts"]
            vals_all = per_sigma_all[sigma]["values"]

            ts_win, vals_win = slice_tail_window_with_fallback(ts_all, vals_all, start_dt, end_dt)
            tail_metrics[(sigma, int(tail_sec))] = snapshot_metrics(vals_win, ts_win)

    row: Dict[str, Any] = {
        "decision_ts_est": decision_dt.isoformat(),
        "next_epoch_time_est": timing.dt_next.isoformat(),
        "curr_epoch": timing.curr_epoch,
        "next_epoch": timing.next_epoch,
        "full_start_est": windows.full_start.isoformat(),
        "full_end_est": windows.full_end.isoformat(),
    }

    for r in registry:
        s = r["sigma"]
        for sec, suffix in [(60, "tail60"), (30, "tail30"), (20, "tail20"), (10, "tail10"), (5, "tail0")]:
            m = tail_metrics.get((s, sec), {})
            row[f"g{s}_{suffix}_last"] = m.get("last")
            row[f"g{s}_{suffix}_delta"] = m.get("delta")
            row[f"g{s}_{suffix}_slope"] = m.get("slope")
            row[f"g{s}_{suffix}_curve"] = m.get("curve")
            row[f"g{s}_{suffix}_tag"] = m.get("tag")

    row["trend"] = trend.trend
    row["confidence"] = trend.confidence
    row["trend_model"] = trend.model
    row["trend_notes"] = trend.notes

    headers = _wide_headers(registry)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def _long_headers() -> List[str]:
    return [
        "decision_ts_est",
        "next_epoch_time_est",
        "curr_epoch",
        "next_epoch",
        "full_start_est",
        "full_end_est",
        "sigma",
        "ts_est",
        "g",
        "window_type",
    ]


def write_long_snapshot(
    config: Dict[str, Any],
    timing: EpochTiming,
    windows: Windows,
    decision_dt: datetime,
    per_sigma_full: Dict[int, Dict[str, Any]],
) -> None:
    out_dir = _ensure_output_dir(config)
    path = out_dir / config["LONG_CSV"]
    file_exists = path.exists()

    headers = _long_headers()
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            w.writeheader()

        common = {
            "decision_ts_est": decision_dt.isoformat(),
            "next_epoch_time_est": timing.dt_next.isoformat(),
            "curr_epoch": timing.curr_epoch,
            "next_epoch": timing.next_epoch,
            "full_start_est": windows.full_start.isoformat(),
            "full_end_est": windows.full_end.isoformat(),
            "window_type": "full",
        }

        for sigma in sorted(per_sigma_full.keys()):
            ts_list = per_sigma_full[sigma].get("ts", [])
            vals = per_sigma_full[sigma].get("values", [])
            n = min(len(ts_list), len(vals))
            for i in range(n):
                t = ts_list[i]
                v = vals[i]
                w.writerow(
                    {
                        **common,
                        "sigma": sigma,
                        "ts_est": t.isoformat() if hasattr(t, "isoformat") else str(t),
                        "g": _to_float_or_none(v),
                    }
                )