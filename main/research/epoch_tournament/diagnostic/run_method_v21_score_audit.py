from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


DEFAULT_INPUT = r"E:/Trading_Bot_V1.0/DogeBets/main/research/epoch_tournament/method_v21/outputs/predictions_v21.csv"
DEFAULT_OUTPUT_DIR = r"/main/research/epoch_tournament/method_v21/audit_outputs"

SCORE_COLS = [
    "v21_bull_continuation_score",
    "v21_bear_continuation_score",
    "v21_bull_exhaustion_score",
    "v21_bear_exhaustion_score",
    "v21_conflict_score",
    "v21_compression_trap_score",
    "v21_bull_invalidators",
    "v21_bear_invalidators",
    "v21_reversal_to_bull_bonus",
    "v21_reversal_to_bear_bonus",
    "v21_bull_raw",
    "v21_bear_raw",
    "v21_neutral_raw",
    "v21_sep",
    "v21_confidence",
]

GATE_DEFAULTS = {
    "direction_min": 0.56,
    "separation_min": 0.10,
    "conflict_neutral": 0.82,
    "conflict_block": 0.74,
    "trap_neutral": 0.78,
    "trap_block": 0.72,
    "continuation_min_soft": 0.48,
    "continuation_min_hard": 0.58,
    "exhaustion_override": 0.62,
    "reversal_promotion": 0.70,
    "fast_slow_warn": 0.52,
    "neutral_close_call": 0.08,
}


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


def _numeric_summary(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"count": 0}
    q = s.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()
    return {
        "count": int(len(s)),
        "min": float(s.min()),
        "mean": float(s.mean()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
        "p01": float(q.get(0.01, float("nan"))),
        "p05": float(q.get(0.05, float("nan"))),
        "p10": float(q.get(0.10, float("nan"))),
        "p25": float(q.get(0.25, float("nan"))),
        "p50": float(q.get(0.50, float("nan"))),
        "p75": float(q.get(0.75, float("nan"))),
        "p90": float(q.get(0.90, float("nan"))),
        "p95": float(q.get(0.95, float("nan"))),
        "p99": float(q.get(0.99, float("nan"))),
        "max": float(s.max()),
    }


def _value_counts(series: pd.Series, top_n: int = 20) -> Dict[str, int]:
    vc = series.fillna("<NA>").astype(str).value_counts().head(top_n)
    return {str(k): int(v) for k, v in vc.items()}


def _gate_audit(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = len(df)

    def pct(mask: pd.Series) -> float:
        return float(mask.fillna(False).mean()) if n else 0.0

    # Core availability
    required = [
        "v21_bull_continuation_score",
        "v21_bear_continuation_score",
        "v21_conflict_score",
        "v21_compression_trap_score",
        "v21_bull_invalidators",
        "v21_bear_invalidators",
        "v21_bull_raw",
        "v21_bear_raw",
        "v21_sep",
    ]
    out["missing_required_cols"] = [c for c in required if c not in df.columns]

    if out["missing_required_cols"]:
        return out

    out["soft_cont_both_fail_rate"] = pct(
        (df["v21_bull_continuation_score"] < GATE_DEFAULTS["continuation_min_soft"]) &
        (df["v21_bear_continuation_score"] < GATE_DEFAULTS["continuation_min_soft"])
    )
    out["hard_bull_fail_rate"] = pct(df["v21_bull_continuation_score"] < GATE_DEFAULTS["continuation_min_hard"])
    out["hard_bear_fail_rate"] = pct(df["v21_bear_continuation_score"] < GATE_DEFAULTS["continuation_min_hard"])

    out["conflict_block_rate"] = pct(df["v21_conflict_score"] >= GATE_DEFAULTS["conflict_block"])
    out["trap_block_rate"] = pct(df["v21_compression_trap_score"] >= GATE_DEFAULTS["trap_block"])

    out["bull_invalidated_rate"] = pct(df["v21_bull_invalidators"] >= 0.62)
    out["bear_invalidated_rate"] = pct(df["v21_bear_invalidators"] >= 0.62)

    out["bull_raw_above_direction_min_rate"] = pct(df["v21_bull_raw"] >= GATE_DEFAULTS["direction_min"])
    out["bear_raw_above_direction_min_rate"] = pct(df["v21_bear_raw"] >= GATE_DEFAULTS["direction_min"])
    out["separation_below_min_rate"] = pct(df["v21_sep"] < GATE_DEFAULTS["separation_min"])

    return out


def _top_examples(df: pd.DataFrame, col: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()

    base_cols = [c for c in ["src_meta_epoch", "v21_trend", "v21_reason", "v21_bull_raw", "v21_bear_raw", "v21_sep"] if c in df.columns]

    # Pull the target column safely even if duplicate labels exist
    series_or_frame = df.loc[:, col]
    if isinstance(series_or_frame, pd.DataFrame):
        series = series_or_frame.iloc[:, 0]
    else:
        series = series_or_frame

    sort_series = pd.to_numeric(series, errors="coerce")

    # Build a clean temporary frame with unique column names
    tmp = df.loc[:, base_cols].copy()
    tmp["__audit_sort_col__"] = sort_series

    tmp = tmp.sort_values("__audit_sort_col__", ascending=ascending).head(n)

    # Put the inspected score back under its original name for readability
    tmp[col] = tmp["__audit_sort_col__"]
    tmp = tmp.drop(columns=["__audit_sort_col__"])

    ordered_cols = [c for c in ["src_meta_epoch", "v21_trend", "v21_reason", col, "v21_bull_raw", "v21_bear_raw", "v21_sep"] if c in tmp.columns]
    return tmp.loc[:, ordered_cols]

def main() -> None:
    ap = argparse.ArgumentParser(description="Audit trend_method_v2_1 score distributions and gate failures.")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_banner("method_v21 Score Audit Starting")
    print(f"[config] input      = {input_path}")
    print(f"[config] output_dir = {output_dir}")
    print("-" * 100)
    print("[stage] loading predictions dataset...")

    t0 = time.time()
    df = _load_frame(input_path)

    dup_counts = df.columns.value_counts()
    dups = dup_counts[dup_counts > 1]
    if len(dups):
        print(f"[warn] duplicate column names detected: {dups.to_dict()}")

    print(f"[data] loaded rows={len(df):,} cols={len(df.columns):,} in {time.time() - t0:.2f}s")

    print("[stage] auditing score distributions...")
    numeric = {}
    present_scores = [c for c in SCORE_COLS if c in df.columns]
    for c in present_scores:
        numeric[c] = _numeric_summary(df[c])

    trend_counts = _value_counts(df["v21_trend"]) if "v21_trend" in df.columns else {}
    reason_counts = _value_counts(df["v21_reason"]) if "v21_reason" in df.columns else {}
    gate_audit = _gate_audit(df)

    report = {
        "input": str(input_path),
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "present_score_cols": present_scores,
        "trend_counts": trend_counts,
        "reason_counts": reason_counts,
        "numeric_summary": numeric,
        "gate_audit": gate_audit,
        "gate_defaults": GATE_DEFAULTS,
    }

    # Save machine-readable report
    json_path = output_dir / "v21_score_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save human-readable report
    txt_path = output_dir / "v21_score_audit.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("method_v21 score audit\n")
        f.write("=" * 80 + "\n")
        f.write(f"input: {input_path}\n")
        f.write(f"rows: {len(df):,}\n")
        f.write(f"cols: {len(df.columns):,}\n\n")

        f.write("trend_counts:\n")
        for k, v in trend_counts.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nreason_counts:\n")
        for k, v in reason_counts.items():
            f.write(f"  {k}: {v}\n")

        f.write("\ngate_audit:\n")
        for k, v in gate_audit.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nnumeric_summary:\n")
        for col, stats in numeric.items():
            f.write(f"\n[{col}]\n")
            for k, v in stats.items():
                f.write(f"  {k}: {v}\n")

    # Save top examples for key columns
    key_cols = [
        "v21_bull_continuation_score",
        "v21_bear_continuation_score",
        "v21_conflict_score",
        "v21_compression_trap_score",
        "v21_bull_invalidators",
        "v21_bear_invalidators",
        "v21_bull_raw",
        "v21_bear_raw",
        "v21_sep",
    ]
    for col in key_cols:
        if col in df.columns:
            hi = _top_examples(df, col, n=args.top_n, ascending=False)
            lo = _top_examples(df, col, n=args.top_n, ascending=True)
            if not hi.empty:
                hi.to_csv(output_dir / f"top_{args.top_n}_{col}_high.csv", index=False)
            if not lo.empty:
                lo.to_csv(output_dir / f"top_{args.top_n}_{col}_low.csv", index=False)

    print("-" * 100)
    print(f"[save] audit_json = {json_path}")
    print(f"[save] audit_txt  = {txt_path}")
    print(f"[model] trend_counts={trend_counts}")
    top_reasons = dict(list(reason_counts.items())[:10])
    print(f"[model] top_reasons={top_reasons}")
    if gate_audit:
        print(f"[audit] gate_summary={gate_audit}")
    _print_banner("method_v21 Score Audit Finished")


if __name__ == "__main__":
    main()
