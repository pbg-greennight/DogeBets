from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

DEFAULT_INPUT = r"E:/Trading_Bot_V1.0/DogeBets/main/data/epoch_model_table_v6/epoch_sequences.parquet"
DEFAULT_OUTPUT_DIR = r"/main/data/epoch_model_table_v6"
DEFAULT_BUILDER = r"E:/Trading_Bot_V1.0/DogeBets/main/research/epoch_tournament/build_v21_src_from_epoch_sequences.py"
DEFAULT_WAKE_OFFSET_SECONDS = 12.0
DEFAULT_MIN_KEEP = 12

SERIES_COLS = [
    "ts_bar_series",
    "open_series",
    "high_series",
    "low_series",
    "close_series",
    "volume_series",
]


def _print_banner(title: str) -> None:
    print("=" * 100)
    print(title)
    print("=" * 100)


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type: {suffix}")


def _load_builder_module(builder_path: Path):
    spec = importlib.util.spec_from_file_location("v21_builder_base", builder_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load builder module from {builder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if hasattr(x, "tolist"):
        vals = x.tolist()
        return vals if isinstance(vals, list) else [vals]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            vals = json.loads(s)
            return vals if isinstance(vals, list) else [vals]
        except Exception:
            return [x]
    return [x]


def _trim_row_to_snapshot(row: pd.Series, wake_offset_seconds: float, min_keep: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    out = row.to_dict()
    ts_vals = _as_list(row.get("ts_bar_series"))
    if not ts_vals:
        return out, {
            "trimmed": False,
            "removed": 0,
            "kept": 0,
            "reason": "missing_ts_bar_series",
        }

    ts = pd.to_datetime(pd.Series(ts_vals), utc=True, errors="coerce")
    valid_mask = ts.notna().to_numpy()
    if not valid_mask.any():
        return out, {
            "trimmed": False,
            "removed": 0,
            "kept": 0,
            "reason": "unparseable_ts_bar_series",
        }

    ts = ts[valid_mask]
    full_n = int(len(ts))

    ts_end_raw = row.get("ts_end")
    ts_end = pd.to_datetime(ts_end_raw, utc=True, errors="coerce")
    if pd.isna(ts_end):
        ts_end = ts.iloc[-1]

    snapshot_cutoff = ts_end - pd.Timedelta(seconds=float(wake_offset_seconds))
    keep_mask_valid = (ts <= snapshot_cutoff).to_numpy()
    keep_count = int(keep_mask_valid.sum())

    if keep_count < min_keep:
        keep_count = min(min_keep, full_n)
        keep_mask_valid = np.zeros(full_n, dtype=bool)
        keep_mask_valid[:keep_count] = True

    kept_positions_valid = np.where(keep_mask_valid)[0]
    if len(kept_positions_valid) == 0:
        keep_count = min(min_keep, full_n)
        keep_mask_valid = np.zeros(full_n, dtype=bool)
        keep_mask_valid[:keep_count] = True
        kept_positions_valid = np.where(keep_mask_valid)[0]

    # Map valid-only positions back to the original ts_bar_series positions.
    orig_positions = np.where(valid_mask)[0][kept_positions_valid]
    orig_keep_mask = np.zeros(len(ts_vals), dtype=bool)
    orig_keep_mask[orig_positions] = True

    removed = int(len(ts_vals) - orig_keep_mask.sum())

    for col in SERIES_COLS:
        vals = _as_list(row.get(col))
        if not vals:
            continue
        n = min(len(vals), len(orig_keep_mask))
        trimmed_vals = [vals[i] for i in range(n) if orig_keep_mask[i]]
        if len(vals) > n:
            # preserve trailing extras only if keeping everything; otherwise drop to stay aligned
            pass
        out[col] = trimmed_vals

    close_vals = _as_list(out.get("close_series"))
    open_vals = _as_list(out.get("open_series"))
    high_vals = _as_list(out.get("high_series"))
    low_vals = _as_list(out.get("low_series"))
    vol_vals = _as_list(out.get("volume_series"))
    ts_kept_vals = _as_list(out.get("ts_bar_series"))

    def _last_float(vals: List[Any], default: Any = None):
        if not vals:
            return default
        try:
            return float(vals[-1])
        except Exception:
            return default

    def _first_float(vals: List[Any], default: Any = None):
        if not vals:
            return default
        try:
            return float(vals[0])
        except Exception:
            return default

    # Preserve full-epoch truth fields, but move meta snapshot fields to the trimmed endpoint.
    out["full_end_close"] = row.get("end_close")
    out["full_price_diff"] = row.get("price_diff")
    out["full_trend_label"] = row.get("trend_label")

    out["ts_snapshot_end"] = ts_kept_vals[-1] if ts_kept_vals else None
    out["bars_snapshot_kept"] = len(close_vals)
    out["bars_snapshot_removed"] = removed
    out["snapshot_cutoff_utc"] = snapshot_cutoff.isoformat()

    if close_vals:
        out["start_close"] = _first_float(close_vals, row.get("start_close"))
        out["end_close"] = _last_float(close_vals, row.get("end_close"))
    if open_vals:
        try:
            out["open_first"] = float(open_vals[0])
        except Exception:
            pass
    if high_vals:
        try:
            out["high_max"] = float(np.nanmax(pd.to_numeric(pd.Series(high_vals), errors="coerce")))
        except Exception:
            pass
    if low_vals:
        try:
            out["low_min"] = float(np.nanmin(pd.to_numeric(pd.Series(low_vals), errors="coerce")))
        except Exception:
            pass
    if vol_vals:
        try:
            vol_num = pd.to_numeric(pd.Series(vol_vals), errors="coerce")
            out["volume_sum"] = float(np.nansum(vol_num))
            out["volume_mean"] = float(np.nanmean(vol_num)) if len(vol_num) else None
        except Exception:
            pass

    return out, {
        "trimmed": removed > 0,
        "removed": removed,
        "kept": len(close_vals),
        "reason": "ok",
    }


def build_snapshot_df(seq_df: pd.DataFrame, wake_offset_seconds: float, min_keep: int) -> tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    trimmed_rows = 0
    removed_total = 0
    kept_total = 0

    for idx, (_, row) in enumerate(seq_df.iterrows(), start=1):
        new_row, info = _trim_row_to_snapshot(row, wake_offset_seconds=wake_offset_seconds, min_keep=min_keep)
        rows.append(new_row)
        trimmed_rows += int(info["trimmed"])
        removed_total += int(info["removed"])
        kept_total += int(info["kept"])
        if idx == 1 or idx % 500 == 0 or idx == len(seq_df):
            print(f"[snapshot_trim] {idx}/{len(seq_df)} epoch={row.get('epoch')} kept={info['kept']} removed={info['removed']}")

    summary = {
        "rows": int(len(rows)),
        "trimmed_rows": int(trimmed_rows),
        "removed_total": int(removed_total),
        "kept_total": int(kept_total),
        "wake_offset_seconds": float(wake_offset_seconds),
        "min_keep": int(min_keep),
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a snapshot-safe v21 source table using epoch data trimmed to decision time.")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--builder", default=DEFAULT_BUILDER)
    ap.add_argument("--wake-offset-seconds", type=float, default=DEFAULT_WAKE_OFFSET_SECONDS)
    ap.add_argument("--min-keep", type=int, default=DEFAULT_MIN_KEEP)
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    builder_path = Path(args.builder)
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_banner("method_v21 Snapshot Feature Factory Starting")
    print(f"[config] input      = {input_path}")
    print(f"[config] output_dir = {output_dir}")
    print(f"[config] builder    = {builder_path}")
    print(f"[config] wake_offset_seconds = {args.wake_offset_seconds}")
    print(f"[config] min_keep   = {args.min_keep}")
    print("-" * 100)

    print("[stage] loading epoch_sequences dataset...")
    t0 = time.time()
    seq_df = _load_frame(input_path)
    print(f"[data] loaded rows={len(seq_df):,} cols={len(seq_df.columns)} in {time.time() - t0:.2f}s")

    print("[stage] trimming sequences to decision snapshot...")
    t1 = time.time()
    snapshot_df, trim_summary = build_snapshot_df(
        seq_df,
        wake_offset_seconds=float(args.wake_offset_seconds),
        min_keep=int(args.min_keep),
    )
    print(f"[snapshot] rows={trim_summary['rows']:,} trimmed_rows={trim_summary['trimmed_rows']:,} removed_total={trim_summary['removed_total']:,} in {time.time() - t1:.2f}s")

    print("[stage] loading base builder module...")
    builder = _load_builder_module(builder_path)

    print("[stage] building snapshot-safe v21 src table...")
    t2 = time.time()
    v21_df = builder.build_v21_src_from_sequences(snapshot_df)
    print(f"[data] built rows={len(v21_df):,} cols={len(v21_df.columns)} in {time.time() - t2:.2f}s")

    parquet_path = output_dir / "epoch_model_table_v21_src_snapshot.parquet"
    summary_path = output_dir / "epoch_model_table_v21_src_snapshot_summary.json"

    v21_df.to_parquet(parquet_path, index=False)

    family_counts = {
        "meta": len([c for c in v21_df.columns if c.startswith("src_meta_")]),
        "hyst": len([c for c in v21_df.columns if c.startswith("src_hyst_")]),
        "msbc": len([c for c in v21_df.columns if c.startswith("src_msbc_")]),
        "gcs": len([c for c in v21_df.columns if c.startswith("src_gcs_")]),
        "dcsd": len([c for c in v21_df.columns if c.startswith("src_dcsd_")]),
        "gbc": len([c for c in v21_df.columns if c.startswith("src_gbc_")]),
    }

    summary = {
        "input": str(input_path),
        "builder": str(builder_path),
        "output_parquet": str(parquet_path),
        "rows": int(len(v21_df)),
        "cols": int(len(v21_df.columns)),
        "family_counts": family_counts,
        "trim_summary": trim_summary,
        "wake_offset_seconds": float(args.wake_offset_seconds),
        "min_keep": int(args.min_keep),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 100)
    print(f"[save] parquet = {parquet_path}")
    print(f"[save] summary = {summary_path}")
    _print_banner("method_v21 Snapshot Feature Factory Finished")


if __name__ == "__main__":
    main()
