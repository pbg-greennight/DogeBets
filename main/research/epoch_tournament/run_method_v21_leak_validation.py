from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd

from method_v21 import apply_trend_method_v2_1, get_v21_config
from build_v21_src_from_epoch_sequences import build_v21_src_from_sequences


STAKE_BNB_DEFAULT = 0.015
BNB_USD_DEFAULT = 685.0
WIN_FEE_RATE_DEFAULT = 0.05
GAS_USD_ROUNDTRIP_DEFAULT = 0.085
DEFAULT_SRC_INPUT = r"E:\Trading_Bot_V1.0\DogeBets\main\data\epoch_model_table_v6\epoch_model_table_v21_src.parquet"
DEFAULT_SEQ_INPUT = r"E:\Trading_Bot_V1.0\DogeBets\main\data\epoch_model_table_v6\epoch_sequences.parquet"


# -----------------------------
# basic logging
# -----------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _banner(title: str) -> None:
    _log("=" * 100)
    _log(title)
    _log("=" * 100)


# -----------------------------
# load / save
# -----------------------------

def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


# -----------------------------
# trend normalization / scoring
# -----------------------------

def _normalize_trend_value(val) -> str | None:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in {"bull", "up", "long", "green", "1"}:
        return "Bull"
    if s in {"bear", "down", "short", "red", "-1"}:
        return "Bear"
    if s in {"neutral", "flat", "skip", "0", "none"}:
        return "Neutral"
    return None


def _guess_actual_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "actual_trend",
        "src_meta_actual_trend",
        "trend_label",
        "target_trend",
        "label",
        "trend_actual",
        "true_trend",
        "actual",
        "winner",
        "outcome",
        "result",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _derive_actual_from_price_diff(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    for col in ["actual_price_diff", "src_meta_actual_price_diff", "price_diff", "src_meta_price_diff"]:
        if col in df.columns:
            out_col = "__derived_actual_trend"
            s = pd.to_numeric(df[col], errors="coerce")
            df[out_col] = s.map(lambda x: "Neutral" if pd.isna(x) or x == 0 else ("Bull" if x > 0 else "Bear"))
            return df, out_col
    return df, None


def _prepare_eval_df(df: pd.DataFrame, actual_col: str | None, pred_col: str = "v21_trend") -> tuple[pd.DataFrame, str | None]:
    work = df.copy()
    effective_actual_col = actual_col if actual_col and actual_col in work.columns else _guess_actual_col(work)
    if effective_actual_col is None:
        work, effective_actual_col = _derive_actual_from_price_diff(work)
    if effective_actual_col is None or pred_col not in work.columns:
        return work.iloc[0:0].copy(), effective_actual_col
    work["__pred_norm"] = work[pred_col].map(_normalize_trend_value)
    work["__actual_norm"] = work[effective_actual_col].map(_normalize_trend_value)
    work = work[work["__pred_norm"].notna() & work["__actual_norm"].notna()].copy()
    return work, effective_actual_col


def _compute_accuracy(eval_df: pd.DataFrame) -> dict[str, Any]:
    if eval_df.empty:
        return {}
    total = len(eval_df)
    correct = int((eval_df["__pred_norm"] == eval_df["__actual_norm"]).sum())
    called = eval_df[eval_df["__pred_norm"] != "Neutral"].copy()
    called_correct = int((called["__pred_norm"] == called["__actual_norm"]).sum())
    return {
        "rows": int(total),
        "correct": correct,
        "wrong": int(total - correct),
        "accuracy": float(correct / total) if total else None,
        "called_rows": int(len(called)),
        "called_correct": int(called_correct),
        "called_wrong": int(len(called) - called_correct),
        "called_accuracy": float(called_correct / len(called)) if len(called) else None,
    }


def _compute_wager(eval_df: pd.DataFrame,
                   stake_bnb: float,
                   bnb_usd: float,
                   win_fee_rate: float,
                   gas_usd_roundtrip: float,
                   bet_on_neutral: bool = False) -> dict[str, Any]:
    if eval_df.empty:
        return {}
    allowed = {"Bull", "Bear", "Neutral"} if bet_on_neutral else {"Bull", "Bear"}
    bets = eval_df[eval_df["__pred_norm"].isin(allowed)].copy()
    if bets.empty:
        return {}
    gas_bnb = gas_usd_roundtrip / bnb_usd if bnb_usd else 0.0
    win_net_bnb = stake_bnb * (1.0 - win_fee_rate) - gas_bnb
    lose_net_bnb = -stake_bnb - gas_bnb
    bets["is_win"] = bets["__pred_norm"] == bets["__actual_norm"]
    bets["pnl_bnb"] = bets["is_win"].map(lambda x: win_net_bnb if x else lose_net_bnb)
    wins = int(bets["is_win"].sum())
    return {
        "bets": int(len(bets)),
        "wins": wins,
        "losses": int(len(bets) - wins),
        "win_rate": float(wins / len(bets)) if len(bets) else None,
        "net_bnb": float(bets["pnl_bnb"].sum()),
        "net_usd": float(bets["pnl_bnb"].sum() * bnb_usd),
    }


# -----------------------------
# tests
# -----------------------------

def _run_model(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    return apply_trend_method_v2_1(df.copy(), config)


def _baseline_test(df: pd.DataFrame,
                   config: dict,
                   actual_col: str | None,
                   stake_bnb: float,
                   bnb_usd: float,
                   win_fee_rate: float,
                   gas_usd_roundtrip: float) -> dict[str, Any]:
    pred = _run_model(df, config)
    eval_df, used_actual = _prepare_eval_df(pred, actual_col)
    return {
        "actual_col_used": used_actual,
        "accuracy": _compute_accuracy(eval_df),
        "wager": _compute_wager(eval_df, stake_bnb, bnb_usd, win_fee_rate, gas_usd_roundtrip),
        "trend_counts": pred["v21_trend"].value_counts(dropna=False).to_dict() if "v21_trend" in pred.columns else {},
    }


def _shuffle_truth_test(df: pd.DataFrame,
                        config: dict,
                        actual_col: str | None,
                        seed: int) -> dict[str, Any]:
    pred = _run_model(df, config)
    eval_df, used_actual = _prepare_eval_df(pred, actual_col)
    if eval_df.empty or used_actual is None:
        return {"actual_col_used": used_actual, "accuracy": {}, "note": "No actual column available."}
    rng = random.Random(seed)
    shuffled = eval_df["__actual_norm"].tolist()
    rng.shuffle(shuffled)
    eval_df = eval_df.copy()
    eval_df["__actual_norm"] = shuffled
    return {"actual_col_used": used_actual, "accuracy": _compute_accuracy(eval_df)}


def _lag_truth_test(df: pd.DataFrame,
                    config: dict,
                    actual_col: str | None,
                    lag: int) -> dict[str, Any]:
    pred = _run_model(df, config)
    eval_df, used_actual = _prepare_eval_df(pred, actual_col)
    if eval_df.empty or used_actual is None:
        return {"actual_col_used": used_actual, "accuracy": {}, "note": "No actual column available."}
    eval_df = eval_df.copy().sort_values("src_meta_epoch" if "src_meta_epoch" in eval_df.columns else eval_df.index.name or eval_df.columns[0])
    eval_df["__actual_norm"] = eval_df["__actual_norm"].shift(lag)
    eval_df = eval_df[eval_df["__actual_norm"].notna()].copy()
    return {"actual_col_used": used_actual, "lag": int(lag), "accuracy": _compute_accuracy(eval_df)}


def _trim_listlike(v, keep_n: int):
    if isinstance(v, list):
        return v[:keep_n]
    if isinstance(v, tuple):
        return list(v[:keep_n])
    return v


def _cutoff_test(seq_df: pd.DataFrame,
                 config: dict,
                 actual_col: str | None,
                 trim_ratio: float,
                 min_keep: int,
                 stake_bnb: float,
                 bnb_usd: float,
                 win_fee_rate: float,
                 gas_usd_roundtrip: float) -> dict[str, Any]:
    work = seq_df.copy()
    series_cols = [
        "ts_bar_series", "open_series", "high_series", "low_series", "close_series", "volume_series"
    ]

    trimmed_rows = []
    trimmed_count = 0
    for _, row in work.iterrows():
        row = row.copy()
        close_series = row.get("close_series")
        if not isinstance(close_series, (list, tuple)) or len(close_series) == 0:
            trimmed_rows.append(row)
            continue
        n = len(close_series)
        keep_n = max(min_keep, int(math.floor(n * (1.0 - trim_ratio))))
        keep_n = min(keep_n, n)
        if keep_n < n:
            trimmed_count += 1
        for c in series_cols:
            row[c] = _trim_listlike(row.get(c), keep_n)
        # recompute epoch-end truth from trimmed sequence for leak-sensitivity test
        trimmed_close = row.get("close_series")
        if isinstance(trimmed_close, list) and trimmed_close:
            start_close = trimmed_close[0]
            end_close = trimmed_close[-1]
            try:
                pdiff = float(end_close) - float(start_close)
            except Exception:
                pdiff = float("nan")
            row["start_close"] = start_close
            row["end_close"] = end_close
            row["price_diff"] = pdiff
            row["trend_label"] = "Neutral" if pd.isna(pdiff) or pdiff == 0 else ("Bull" if pdiff > 0 else "Bear")
        trimmed_rows.append(row)

    trimmed_seq_df = pd.DataFrame(trimmed_rows)
    rebuilt_src = build_v21_src_from_sequences(trimmed_seq_df)
    pred = _run_model(rebuilt_src, config)
    eval_df, used_actual = _prepare_eval_df(pred, actual_col)
    return {
        "actual_col_used": used_actual,
        "trim_ratio": float(trim_ratio),
        "min_keep": int(min_keep),
        "trimmed_rows": int(trimmed_count),
        "rebuilt_rows": int(len(rebuilt_src)),
        "accuracy": _compute_accuracy(eval_df),
        "wager": _compute_wager(eval_df, stake_bnb, bnb_usd, win_fee_rate, gas_usd_roundtrip),
    }


# -----------------------------
# reporting
# -----------------------------

def _risk_flag(baseline: dict[str, Any], shuffle: dict[str, Any], lag: dict[str, Any], cutoff: dict[str, Any] | None) -> dict[str, Any]:
    base_called = ((baseline.get("accuracy") or {}).get("called_accuracy")) or 0.0
    shuf_called = ((shuffle.get("accuracy") or {}).get("called_accuracy")) or 0.0
    lag_called = ((lag.get("accuracy") or {}).get("called_accuracy")) or 0.0
    cutoff_called = (((cutoff or {}).get("accuracy") or {}).get("called_accuracy")) if cutoff else None

    flags = []
    if shuf_called > 0.56:
        flags.append("shuffle_called_accuracy_too_high")
    if lag_called > 0.56:
        flags.append("lag_called_accuracy_too_high")
    if cutoff_called is not None and cutoff_called >= base_called:
        flags.append("cutoff_test_not_lower_than_baseline")

    verdict = "PASS" if not flags else "REVIEW"
    return {
        "verdict": verdict,
        "baseline_called_accuracy": base_called,
        "shuffle_called_accuracy": shuf_called,
        "lag_called_accuracy": lag_called,
        "cutoff_called_accuracy": cutoff_called,
        "flags": flags,
    }


def _write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    json_path = out_dir / "leak_validation_v21.json"
    txt_path = out_dir / "leak_validation_v21.txt"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = []
    lines.append("trend_method_v2_1 leak validation")
    lines.append("=" * 40)
    risk = payload.get("risk_check") or {}
    lines.append(f"verdict: {risk.get('verdict')}")
    lines.append(f"flags: {risk.get('flags')}")
    lines.append("")
    for key in ["baseline", "shuffle_truth", "lag_truth", "cutoff_trim"]:
        block = payload.get(key)
        if not block:
            continue
        lines.append(key)
        lines.append("-" * 40)
        lines.append(f"actual_col_used: {block.get('actual_col_used')}")
        if "trim_ratio" in block:
            lines.append(f"trim_ratio: {block.get('trim_ratio')} | min_keep: {block.get('min_keep')} | trimmed_rows: {block.get('trimmed_rows')}")
        if "lag" in block:
            lines.append(f"lag: {block.get('lag')}")
        acc = block.get("accuracy") or {}
        if acc:
            lines.append(
                f"accuracy: total={acc.get('accuracy')} ({acc.get('correct')}/{acc.get('rows')}) | called={acc.get('called_accuracy')} ({acc.get('called_correct')}/{acc.get('called_rows')})"
            )
        wager = block.get("wager") or {}
        if wager:
            lines.append(
                f"wager: bets={wager.get('bets')} | win_rate={wager.get('win_rate')} | net_bnb={wager.get('net_bnb')} | net_usd={wager.get('net_usd')}"
            )
        lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# cli
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="One-command leak validation for trend_method_v2_1.")
    ap.add_argument("--src-input", default=DEFAULT_SRC_INPUT, help="Canonical v21 src parquet/csv used by the research runner.")
    ap.add_argument("--seq-input", default=DEFAULT_SEQ_INPUT, help="Epoch sequences parquet/csv for optional cutoff rebuild test.")
    ap.add_argument("--output-dir", default=None, help="Directory to save leak validation outputs. Defaults next to src input.")
    ap.add_argument("--actual-col", default=None, help="Optional actual trend column name.")
    ap.add_argument("--limit", type=int, default=None, help="Optional row limit for quick testing.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for shuffle test.")
    ap.add_argument("--lag", type=int, default=1, help="Lag for lag-truth test.")
    ap.add_argument("--trim-ratio", type=float, default=0.10, help="Cutoff test trim ratio from end of each epoch sequence.")
    ap.add_argument("--min-keep", type=int, default=12, help="Minimum bars to retain per trimmed epoch.")
    ap.add_argument("--skip-cutoff", action="store_true", help="Skip the expensive cutoff/rebuild test.")
    ap.add_argument("--stake-bnb", type=float, default=STAKE_BNB_DEFAULT)
    ap.add_argument("--bnb-usd", type=float, default=BNB_USD_DEFAULT)
    ap.add_argument("--win-fee-rate", type=float, default=WIN_FEE_RATE_DEFAULT, help="Fee rate on winnings, e.g. 0.05 for 5%%.")
    ap.add_argument("--gas-usd-roundtrip", type=float, default=GAS_USD_ROUNDTRIP_DEFAULT)
    args = ap.parse_args()

    src_input = Path(args.src_input)
    seq_input = Path(args.seq_input)
    out_dir = Path(args.output_dir) if args.output_dir else (src_input.parent / "leak_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    _banner("trend_method_v2_1 Leak Validation Starting")
    _log(f"[config] src_input   = {src_input}")
    _log(f"[config] seq_input   = {seq_input}")
    _log(f"[config] output_dir  = {out_dir}")
    _log(f"[config] actual_col  = {args.actual_col}")
    _log(f"[config] limit       = {args.limit}")
    _log(f"[config] lag         = {args.lag}")
    _log(f"[config] trim_ratio  = {args.trim_ratio}")
    _log(f"[config] min_keep    = {args.min_keep}")
    _log(f"[config] skip_cutoff = {args.skip_cutoff}")
    _log("-" * 100)

    t0 = time.perf_counter()
    config = get_v21_config()

    src_df = _load_frame(src_input)
    if args.limit is not None:
        src_df = src_df.head(args.limit).copy()
    _log(f"[data] src rows={len(src_df):,} cols={len(src_df.columns):,}")

    _log("[test] baseline ...")
    baseline = _baseline_test(src_df, config, args.actual_col, args.stake_bnb, args.bnb_usd, args.win_fee_rate, args.gas_usd_roundtrip)
    _log(f"[baseline] called_accuracy={((baseline.get('accuracy') or {}).get('called_accuracy'))} | net_bnb={((baseline.get('wager') or {}).get('net_bnb'))}")

    _log("[test] shuffle_truth ...")
    shuffle = _shuffle_truth_test(src_df, config, args.actual_col, args.seed)
    _log(f"[shuffle] called_accuracy={((shuffle.get('accuracy') or {}).get('called_accuracy'))}")

    _log("[test] lag_truth ...")
    lag = _lag_truth_test(src_df, config, args.actual_col, args.lag)
    _log(f"[lag] called_accuracy={((lag.get('accuracy') or {}).get('called_accuracy'))}")

    cutoff = None
    if not args.skip_cutoff:
        _log("[test] cutoff_trim ...")
        seq_df = _load_frame(seq_input)
        if args.limit is not None:
            seq_df = seq_df.head(args.limit).copy()
        cutoff = _cutoff_test(seq_df, config, args.actual_col, args.trim_ratio, args.min_keep, args.stake_bnb, args.bnb_usd, args.win_fee_rate, args.gas_usd_roundtrip)
        _log(f"[cutoff] called_accuracy={((cutoff.get('accuracy') or {}).get('called_accuracy'))} | net_bnb={((cutoff.get('wager') or {}).get('net_bnb'))}")

    risk = _risk_flag(baseline, shuffle, lag, cutoff)
    _log("-" * 100)
    _log(f"[risk_check] verdict={risk.get('verdict')} | flags={risk.get('flags')}")

    payload = {
        "config": {
            "src_input": str(src_input),
            "seq_input": str(seq_input),
            "actual_col_requested": args.actual_col,
            "limit": args.limit,
            "lag": args.lag,
            "trim_ratio": args.trim_ratio,
            "min_keep": args.min_keep,
            "stake_bnb": args.stake_bnb,
            "bnb_usd": args.bnb_usd,
            "win_fee_rate": args.win_fee_rate,
            "gas_usd_roundtrip": args.gas_usd_roundtrip,
            "skip_cutoff": args.skip_cutoff,
        },
        "baseline": baseline,
        "shuffle_truth": shuffle,
        "lag_truth": lag,
        "cutoff_trim": cutoff,
        "risk_check": risk,
    }
    _write_outputs(out_dir, payload)

    _banner("trend_method_v2_1 Leak Validation Finished")
    _log(f"[save] json = {out_dir / 'leak_validation_v21.json'}")
    _log(f"[save] txt  = {out_dir / 'leak_validation_v21.txt'}")
    _log(f"[done] total_time = {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
