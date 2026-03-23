from __future__ import annotations

import argparse
import copy
import warnings
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from main.research.epoch_tournament.method_v21 import apply_trend_method_v2_1, get_v21_config


DEFAULT_INPUT = r"E:/Trading_Bot_V1.0/DogeBets/main/research/epoch_tournament/method_v21/outputs/predictions_v21.csv"
DEFAULT_OUTPUT_DIR = r"E:/Trading_Bot_V1.0/DogeBets/main/research/epoch_tournament/method_v21/sweep_outputs"
DEFAULT_PREDICTIONS_NAME = "predictions_v21_sweep_best.csv"

SWEEP_KEYS = [
    "direction_min",
    "continuation_min_soft",
    "continuation_min_hard",
    "separation_min",
]




def _configure_runtime_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Downcasting object dtype arrays on \.fillna, \.ffill, \.bfill is deprecated.*",
        category=FutureWarning,
    )

def _log(msg: str) -> None:
    print(msg, flush=True)


def _banner(title: str) -> None:
    _log("=" * 100)
    _log(title)
    _log("=" * 100)


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}. Use CSV or parquet.")


def _save_frame(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {path.suffix}. Use CSV or parquet.")


def _parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Parsed empty float list from sweep argument.")
    return out


def _get_default_input_from_config(config: Dict[str, Any]) -> str | None:
    research_cfg = config.get("research", {}) if isinstance(config, dict) else {}
    return research_cfg.get("default_input")


def _get_default_output_dir_from_config(config: Dict[str, Any]) -> str | None:
    research_cfg = config.get("research", {}) if isinstance(config, dict) else {}
    return research_cfg.get("default_output_dir")


def _source_schema_diagnostics(df: pd.DataFrame) -> dict:
    src_cols = [c for c in df.columns if c.startswith("src_")]
    fams = {
        "meta": [c for c in src_cols if c.startswith("src_meta_")],
        "hyst": [c for c in src_cols if c.startswith("src_hyst_")],
        "msbc": [c for c in src_cols if c.startswith("src_msbc_")],
        "gcs": [c for c in src_cols if c.startswith("src_gcs_")],
        "dcsd": [c for c in src_cols if c.startswith("src_dcsd_") or c.startswith("src_csd_")],
        "gbc": [c for c in src_cols if c.startswith("src_gbc_")],
    }
    return {
        "src_total": len(src_cols),
        "family_counts": {k: len(v) for k, v in fams.items()},
    }


def _compute_basic_summary(df: pd.DataFrame, actual_col: str | None) -> dict:
    summary: dict = {
        "rows": int(len(df)),
        "trend_counts": df["v21_trend"].value_counts(dropna=False).to_dict() if "v21_trend" in df.columns else {},
        "reason_counts": df["v21_reason"].value_counts(dropna=False).head(20).to_dict() if "v21_reason" in df.columns else {},
        "confidence_mean": float(df["v21_confidence"].mean()) if "v21_confidence" in df.columns and len(df) else None,
        "coverage_non_neutral": float((df["v21_trend"] != "Neutral").mean()) if "v21_trend" in df.columns and len(df) else None,
    }

    if actual_col and actual_col in df.columns and "v21_trend" in df.columns:
        eval_df = df[df[actual_col].notna()].copy()
        if len(eval_df):
            summary["actual_col"] = actual_col
            summary["overall_accuracy"] = float((eval_df["v21_trend"] == eval_df[actual_col]).mean())
            non_neutral = eval_df[eval_df["v21_trend"] != "Neutral"]
            summary["called_rows"] = int(len(non_neutral))
            summary["called_accuracy"] = float((non_neutral["v21_trend"] == non_neutral[actual_col]).mean()) if len(non_neutral) else None
            if len(non_neutral):
                by_pred = {}
                for trend, sub in non_neutral.groupby("v21_trend"):
                    by_pred[str(trend)] = {
                        "rows": int(len(sub)),
                        "accuracy": float((sub["v21_trend"] == sub[actual_col]).mean()),
                    }
                summary["by_predicted_trend"] = by_pred
    return summary


def _score_row(summary: Dict[str, Any]) -> float:
    cov = float(summary.get("coverage_non_neutral") or 0.0)
    called_acc = summary.get("called_accuracy")
    if called_acc is None:
        called_acc = 0.0
    overall_acc = summary.get("overall_accuracy")
    if overall_acc is None:
        overall_acc = 0.0

    # Coverage is crucial here because the current failure mode is all-neutral.
    return (0.60 * float(called_acc)) + (0.25 * cov) + (0.15 * float(overall_acc))


def _flatten_key_paths(obj: Any, prefix: Tuple[str, ...] = ()) -> List[Tuple[str, ...]]:
    paths: List[Tuple[str, ...]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = prefix + (str(k),)
            paths.append(new_prefix)
            paths.extend(_flatten_key_paths(v, new_prefix))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_prefix = prefix + (str(i),)
            paths.append(new_prefix)
            paths.extend(_flatten_key_paths(v, new_prefix))
    return paths


def _set_by_path(obj: Any, path: Tuple[str, ...], value: Any) -> bool:
    cur = obj
    for token in path[:-1]:
        if isinstance(cur, dict):
            if token not in cur:
                return False
            cur = cur[token]
        elif isinstance(cur, list):
            try:
                idx = int(token)
            except Exception:
                return False
            if idx < 0 or idx >= len(cur):
                return False
            cur = cur[idx]
        else:
            return False

    last = path[-1]
    if isinstance(cur, dict):
        cur[last] = value
        return True
    if isinstance(cur, list):
        try:
            idx = int(last)
        except Exception:
            return False
        if idx < 0 or idx >= len(cur):
            return False
        cur[idx] = value
        return True
    return False


def _find_matching_paths(config: Dict[str, Any], key_name: str) -> List[Tuple[str, ...]]:
    matches: List[Tuple[str, ...]] = []
    for path in _flatten_key_paths(config):
        if path and path[-1] == key_name:
            matches.append(path)
    return matches


def _inject_thresholds(config: Dict[str, Any], updates: Dict[str, float]) -> Dict[str, List[str]]:
    touched: Dict[str, List[str]] = {}
    for key_name, value in updates.items():
        matches = _find_matching_paths(config, key_name)
        if not matches:
            continue
        touched[key_name] = []
        for path in matches:
            ok = _set_by_path(config, path, value)
            if ok:
                touched[key_name].append(".".join(path))
    return touched


def _collect_result_row(summary: Dict[str, Any], params: Dict[str, float], elapsed_s: float, touched: Dict[str, List[str]]) -> Dict[str, Any]:
    trend_counts = summary.get("trend_counts") or {}
    row: Dict[str, Any] = {
        **params,
        "rows": int(summary.get("rows", 0)),
        "bull_calls": int(trend_counts.get("Bull", 0)),
        "bear_calls": int(trend_counts.get("Bear", 0)),
        "neutral_calls": int(trend_counts.get("Neutral", 0)),
        "coverage_non_neutral": summary.get("coverage_non_neutral"),
        "called_rows": summary.get("called_rows", 0),
        "called_accuracy": summary.get("called_accuracy"),
        "overall_accuracy": summary.get("overall_accuracy"),
        "confidence_mean": summary.get("confidence_mean"),
        "rank_score": _score_row(summary),
        "elapsed_s": elapsed_s,
        "touch_map": json.dumps(touched, ensure_ascii=False),
    }
    reason_counts = summary.get("reason_counts") or {}
    row["top_reason_1"] = next(iter(reason_counts.keys()), None)
    row["top_reason_1_count"] = next(iter(reason_counts.values()), None)
    return row


def _is_valid_candidate(row: Dict[str, Any], min_coverage: float) -> bool:
    cov = row.get("coverage_non_neutral")
    if cov is None or cov < min_coverage:
        return False
    if row.get("bull_calls", 0) <= 0:
        return False
    if row.get("bear_calls", 0) <= 0:
        return False
    return True


def _write_summary_file(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _configure_runtime_warnings()
    base_config = get_v21_config()
    default_input = _get_default_input_from_config(base_config) or DEFAULT_INPUT
    default_output_dir = _get_default_output_dir_from_config(base_config) or DEFAULT_OUTPUT_DIR

    ap = argparse.ArgumentParser(description="Run a parameter sweep over method_v21 decision thresholds.")
    ap.add_argument("--input", default=default_input, help="Input CSV or parquet with canonical src_* columns or already-mapped columns.")
    ap.add_argument("--output-dir", default=default_output_dir, help="Directory to save sweep outputs.")
    ap.add_argument("--limit", type=int, default=None, help="Optional row limit for quick testing.")
    ap.add_argument("--actual-col", default=None, help="Optional actual trend column for accuracy metrics.")
    ap.add_argument("--predictions-name", default=DEFAULT_PREDICTIONS_NAME, help="Best-run predictions output filename (.csv or .parquet).")
    ap.add_argument("--direction-min", default="0.30,0.34,0.38,0.42,0.46,0.50,0.54", help="Comma-separated sweep values.")
    ap.add_argument("--continuation-min-soft", default="0.28,0.32,0.36,0.40,0.44,0.48", help="Comma-separated sweep values.")
    ap.add_argument("--continuation-min-hard", default="0.34,0.38,0.42,0.46,0.50,0.54", help="Comma-separated sweep values.")
    ap.add_argument("--separation-min", default="0.02,0.04,0.06,0.08,0.10", help="Comma-separated sweep values.")
    ap.add_argument("--min-coverage", type=float, default=0.10, help="Minimum non-neutral coverage to count as a valid candidate.")
    ap.add_argument("--top-k", type=int, default=25, help="How many top rows to save to top-results files.")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    direction_vals = _parse_float_list(args.direction_min)
    cont_soft_vals = _parse_float_list(args.continuation_min_soft)
    cont_hard_vals = _parse_float_list(args.continuation_min_hard)
    sep_vals = _parse_float_list(args.separation_min)

    combos = list(itertools.product(direction_vals, cont_soft_vals, cont_hard_vals, sep_vals))

    _banner("method_v21 Parameter Sweep Starting")
    _log(f"[config] input      = {input_path}")
    _log(f"[config] output_dir = {out_dir}")
    _log(f"[config] actual_col = {args.actual_col}")
    _log(f"[config] limit      = {args.limit}")
    _log(f"[sweep] direction_min         = {direction_vals}")
    _log(f"[sweep] continuation_min_soft = {cont_soft_vals}")
    _log(f"[sweep] continuation_min_hard = {cont_hard_vals}")
    _log(f"[sweep] separation_min        = {sep_vals}")
    _log(f"[sweep] combinations          = {len(combos):,}")
    _log("-" * 100)

    t0 = time.perf_counter()
    _log("[stage] loading archived dataset...")
    df = _load_frame(input_path)
    _log(f"[data] loaded rows={len(df):,} cols={len(df.columns):,} in {(time.perf_counter() - t0):.2f}s")

    if args.limit is not None:
        df = df.head(args.limit).copy()
        _log(f"[data] applied limit -> rows={len(df):,}")

    diag = _source_schema_diagnostics(df)
    _log(f"[schema] src_total={diag['src_total']} | family_counts={diag['family_counts']}")
    _log("-" * 100)

    results: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None
    best_pred_df: pd.DataFrame | None = None
    best_params: Dict[str, float] | None = None
    best_touch_map: Dict[str, List[str]] | None = None

    sweep_t0 = time.perf_counter()
    for idx, (direction_min, cont_soft, cont_hard, sep_min) in enumerate(combos, start=1):
        params = {
            "direction_min": float(direction_min),
            "continuation_min_soft": float(cont_soft),
            "continuation_min_hard": float(cont_hard),
            "separation_min": float(sep_min),
        }

        cfg = copy.deepcopy(base_config)
        touched = _inject_thresholds(cfg, params)
        missing_keys = [k for k in SWEEP_KEYS if k not in touched]
        if missing_keys:
            raise KeyError(
                "Unable to locate sweep keys in get_v21_config(): "
                f"{missing_keys}. Located keys were: {sorted(set(p[-1] for p in _flatten_key_paths(base_config)))}"
            )

        run_t0 = time.perf_counter()
        pred_df = apply_trend_method_v2_1(df, cfg)
        run_elapsed = time.perf_counter() - run_t0
        summary = _compute_basic_summary(pred_df, args.actual_col)
        row = _collect_result_row(summary, params, run_elapsed, touched)
        results.append(row)

        if best_row is None or float(row["rank_score"]) > float(best_row["rank_score"]):
            best_row = row
            best_pred_df = pred_df.copy()
            best_params = params.copy()
            best_touch_map = copy.deepcopy(touched)

        if idx == 1 or idx % 25 == 0 or idx == len(combos):
            _log(
                f"[progress] {idx:,}/{len(combos):,} | "
                f"best_score={best_row['rank_score']:.6f} | "
                f"last_cov={row['coverage_non_neutral']} | "
                f"last_called_acc={row['called_accuracy']}"
            )

    sweep_elapsed = time.perf_counter() - sweep_t0
    _log("-" * 100)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        ["rank_score", "called_accuracy", "coverage_non_neutral", "overall_accuracy"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    valid_df = result_df[result_df.apply(lambda r: _is_valid_candidate(r.to_dict(), args.min_coverage), axis=1)].copy()
    valid_df = valid_df.reset_index(drop=True)

    all_csv = out_dir / "v21_param_sweep_all.csv"
    valid_csv = out_dir / "v21_param_sweep_valid.csv"
    top_csv = out_dir / f"v21_param_sweep_top_{args.top_k}.csv"
    best_json = out_dir / "v21_param_sweep_best.json"
    summary_txt = out_dir / "v21_param_sweep_summary.txt"

    result_df.to_csv(all_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    result_df.head(args.top_k).to_csv(top_csv, index=False)

    if best_row is None or best_pred_df is None or best_params is None or best_touch_map is None:
        raise RuntimeError("Sweep produced no results.")

    best_predictions_path = out_dir / args.predictions_name
    _save_frame(best_pred_df, best_predictions_path)

    best_payload = {
        "input": str(input_path),
        "output_dir": str(out_dir),
        "sweep_elapsed_s": sweep_elapsed,
        "combinations": len(combos),
        "best_params": best_params,
        "best_touch_map": best_touch_map,
        "best_result": best_row,
        "valid_candidates": int(len(valid_df)),
        "all_results_csv": str(all_csv),
        "valid_results_csv": str(valid_csv),
        "top_results_csv": str(top_csv),
        "best_predictions": str(best_predictions_path),
    }
    best_json.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    lines = [
        "method_v21 parameter sweep summary",
        "=" * 60,
        f"input: {input_path}",
        f"output_dir: {out_dir}",
        f"rows: {len(df):,}",
        f"combinations: {len(combos):,}",
        f"sweep_elapsed_s: {sweep_elapsed:.2f}",
        f"valid_candidates: {len(valid_df):,}",
        "",
        "best_params:",
    ]
    for k, v in best_params.items():
        lines.append(f"  {k}: {v}")
    lines.extend([
        "",
        "best_result:",
    ])
    for k, v in best_row.items():
        lines.append(f"  {k}: {v}")
    lines.extend([
        "",
        f"all_results_csv: {all_csv}",
        f"valid_results_csv: {valid_csv}",
        f"top_results_csv: {top_csv}",
        f"best_predictions: {best_predictions_path}",
        f"best_json: {best_json}",
    ])
    _write_summary_file(summary_txt, lines)

    _log(f"[save] all_results_csv   = {all_csv}")
    _log(f"[save] valid_results_csv = {valid_csv}")
    _log(f"[save] top_results_csv   = {top_csv}")
    _log(f"[save] best_json         = {best_json}")
    _log(f"[save] best_summary_txt  = {summary_txt}")
    _log(f"[save] best_predictions  = {best_predictions_path}")
    _log(f"[best] params            = {best_params}")
    _log(f"[best] score             = {best_row['rank_score']}")
    _log(f"[best] coverage          = {best_row['coverage_non_neutral']}")
    _log(f"[best] called_accuracy   = {best_row['called_accuracy']}")
    _log(f"[best] overall_accuracy  = {best_row['overall_accuracy']}")

    _banner("method_v21 Parameter Sweep Finished")
    _log(f"[done] total_time = {(time.perf_counter() - t0):.2f}s")


if __name__ == "__main__":
    main()
