from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

from method_v21 import apply_trend_method_v2_1, get_v21_config


STAKE_BNB_DEFAULT = 0.015
BNB_USD_DEFAULT = 685.0
WIN_FEE_RATE_DEFAULT = 0.05
GAS_USD_ROUNDTRIP_DEFAULT = 0.085

CONF_LOW_GRID_DEFAULT = [0.58, 0.59, 0.60, 0.61]
CONF_HIGH_GRID_DEFAULT = [0.62, 0.64, 0.66]
LOW_SEP_MAX_GRID_DEFAULT = [0.01, 0.015, 0.02]
HIGH_SEP_MIN_GRID_DEFAULT = [0.035, 0.04, 0.045, 0.05]


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


def _source_schema_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
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


def _guess_actual_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "actual_trend",
        "src_meta_actual_trend",
        "trend_label",
        "actual_label",
        "label",
        "target",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _norm_trend(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in {"bull", "up", "long", "buy", "1"}:
        return "Bull"
    if s in {"bear", "down", "short", "sell", "-1"}:
        return "Bear"
    if s in {"neutral", "flat", "0"}:
        return "Neutral"
    return str(v)


def _compute_actual_context(df: pd.DataFrame, actual_col: str) -> pd.DataFrame:
    out = df.copy()
    out[actual_col] = out[actual_col].map(_norm_trend)
    directional = out[actual_col].where(out[actual_col].isin(["Bull", "Bear"]))
    prev_directional = directional.ffill().shift(1)
    out["actual_dir_context"] = directional
    out["actual_prev_dir_context"] = prev_directional
    out["actual_phase"] = None
    mask = directional.notna() & prev_directional.notna()
    out.loc[mask, "actual_phase"] = (directional[mask] == prev_directional[mask]).map(
        lambda x: "run" if bool(x) else "flip"
    )
    return out


def _apply_selective_rollback(
    pred_df: pd.DataFrame,
    conf_low: float,
    conf_high: float,
    low_sep_max: float,
    high_sep_min: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pred_df.copy()
    df["v21_3_rollback_applied"] = 0
    df["v21_3_prev_trend"] = df["v21_trend"]
    df["v21_3_prev_reason"] = df["v21_reason"]

    required = {"v21_trend", "v21_reason", "v21_confidence", "v21_sep", "v21_bull_raw", "v21_bear_raw"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for rollback: {missing}")

    conf_mask = (df["v21_confidence"] >= conf_low) & (df["v21_confidence"] <= conf_high)
    sep_mask = (df["v21_sep"] < low_sep_max) | (df["v21_sep"] >= high_sep_min)
    reason_mask = df["v21_reason"] == "NEU_WEAK_CONTINUATION_BOTH"
    trend_mask = df["v21_trend"] == "Neutral"
    rb_mask = trend_mask & reason_mask & conf_mask & sep_mask

    stronger_bull = df["v21_bull_raw"] >= df["v21_bear_raw"]
    forced_side = stronger_bull.map(lambda x: "Bull" if bool(x) else "Bear")
    df.loc[rb_mask, "v21_trend"] = forced_side[rb_mask]
    df.loc[rb_mask, "v21_reason"] = "ROLLBACK_WEAK_CONT_BOTH"
    df.loc[rb_mask, "v21_3_rollback_applied"] = 1

    rollback_rows = df.loc[rb_mask].copy()
    rollback_rows["rollback_forced_side"] = forced_side[rb_mask]
    return df, rollback_rows


def _compute_wager_stats(
    eval_df: pd.DataFrame,
    actual_col: str,
    stake_bnb: float,
    bnb_usd: float,
    win_fee_rate: float,
    gas_usd_roundtrip: float,
    bet_neutral: bool = False,
) -> dict[str, Any]:
    bet_mask = eval_df["v21_trend"].isin(["Bull", "Bear"]) if not bet_neutral else eval_df["v21_trend"].isin(["Bull", "Bear", "Neutral"])
    bets_df = eval_df.loc[bet_mask].copy()
    gas_bnb = gas_usd_roundtrip / bnb_usd
    per_win_bnb = (stake_bnb * (1.0 - win_fee_rate)) - gas_bnb
    per_loss_bnb = -stake_bnb - gas_bnb

    wins = int((bets_df["v21_trend"] == bets_df[actual_col]).sum())
    losses = int(len(bets_df) - wins)
    net_bnb = wins * per_win_bnb + losses * per_loss_bnb
    return {
        "bets": int(len(bets_df)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(bets_df)) if len(bets_df) else None,
        "net_bnb": float(net_bnb),
        "net_usd": float(net_bnb * bnb_usd),
        "per_win_bnb": float(per_win_bnb),
        "per_loss_bnb": float(per_loss_bnb),
        "fee_on_wins": float(win_fee_rate),
        "gas_usd_roundtrip": float(gas_usd_roundtrip),
    }


def _pred_stats(eval_df: pd.DataFrame, actual_col: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["overall_accuracy"] = float((eval_df["v21_trend"] == eval_df[actual_col]).mean()) if len(eval_df) else None
    called = eval_df[eval_df["v21_trend"].isin(["Bull", "Bear"])].copy()
    out["called_rows"] = int(len(called))
    out["called_accuracy"] = float((called["v21_trend"] == called[actual_col]).mean()) if len(called) else None

    for trend in ["Bull", "Bear", "Neutral"]:
        sub = eval_df[eval_df["v21_trend"] == trend].copy()
        correct = int((sub["v21_trend"] == sub[actual_col]).sum()) if len(sub) else 0
        out[f"pred_{trend.lower()}_rows"] = int(len(sub))
        out[f"pred_{trend.lower()}_correct"] = correct
        out[f"pred_{trend.lower()}_wrong"] = int(len(sub) - correct)
        out[f"pred_{trend.lower()}_acc"] = float(correct / len(sub)) if len(sub) else None
        if trend in {"Bull", "Bear"}:
            for phase in ["run", "flip"]:
                phase_sub = sub[sub["actual_phase"] == phase].copy()
                phase_correct = int((phase_sub["v21_trend"] == phase_sub[actual_col]).sum()) if len(phase_sub) else 0
                out[f"pred_{trend.lower()}_{phase}_rows"] = int(len(phase_sub))
                out[f"pred_{trend.lower()}_{phase}_correct"] = phase_correct
                out[f"pred_{trend.lower()}_{phase}_wrong"] = int(len(phase_sub) - phase_correct)
                out[f"pred_{trend.lower()}_{phase}_acc"] = float(phase_correct / len(phase_sub)) if len(phase_sub) else None
    return out


def _forced_remaining_false_neutral(remaining_fn_df: pd.DataFrame, actual_col: str, stake_bnb: float, bnb_usd: float, win_fee_rate: float, gas_usd_roundtrip: float) -> dict[str, Any]:
    if len(remaining_fn_df) == 0:
        return {"rows": 0}
    gas_bnb = gas_usd_roundtrip / bnb_usd
    per_win_bnb = (stake_bnb * (1.0 - win_fee_rate)) - gas_bnb
    per_loss_bnb = -stake_bnb - gas_bnb
    stronger_bull = remaining_fn_df["v21_bull_raw"] >= remaining_fn_df["v21_bear_raw"]
    forced_side = stronger_bull.map(lambda x: "Bull" if bool(x) else "Bear")
    wins = int((forced_side == remaining_fn_df[actual_col]).sum())
    losses = int(len(remaining_fn_df) - wins)
    net_bnb = wins * per_win_bnb + losses * per_loss_bnb
    return {
        "rows": int(len(remaining_fn_df)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(remaining_fn_df)) if len(remaining_fn_df) else None,
        "net_bnb": float(net_bnb),
        "net_usd": float(net_bnb * bnb_usd),
        "side_counts": forced_side.value_counts(dropna=False).to_dict(),
    }


def _reason_leaderboard(remaining_fn_df: pd.DataFrame, actual_col: str, stake_bnb: float, bnb_usd: float, win_fee_rate: float, gas_usd_roundtrip: float) -> pd.DataFrame:
    if len(remaining_fn_df) == 0:
        return pd.DataFrame()
    gas_bnb = gas_usd_roundtrip / bnb_usd
    per_win_bnb = (stake_bnb * (1.0 - win_fee_rate)) - gas_bnb
    per_loss_bnb = -stake_bnb - gas_bnb
    rows = []
    for reason, sub in remaining_fn_df.groupby("v21_reason", dropna=False):
        stronger_bull = sub["v21_bull_raw"] >= sub["v21_bear_raw"]
        forced_side = stronger_bull.map(lambda x: "Bull" if bool(x) else "Bear")
        wins = int((forced_side == sub[actual_col]).sum())
        losses = int(len(sub) - wins)
        net_bnb = wins * per_win_bnb + losses * per_loss_bnb
        rows.append({
            "reason": str(reason),
            "rows": int(len(sub)),
            "forced_side_rows": int(len(sub)),
            "forced_side_wins": wins,
            "forced_side_losses": losses,
            "forced_side_win_rate": float(wins / len(sub)) if len(sub) else None,
            "forced_side_net_bnb": float(net_bnb),
            "forced_side_net_usd": float(net_bnb * bnb_usd),
            "avg_confidence": float(sub["v21_confidence"].mean()) if "v21_confidence" in sub.columns else None,
            "avg_sep": float(sub["v21_sep"].mean()) if "v21_sep" in sub.columns else None,
            "avg_abs_price_diff": float(sub["actual_price_diff"].abs().mean()) if "actual_price_diff" in sub.columns else None,
        })
    return pd.DataFrame(rows).sort_values(["forced_side_net_bnb", "forced_side_win_rate", "rows"], ascending=[False, False, False])


def _weak_bands(remaining_fn_df: pd.DataFrame, actual_col: str, stake_bnb: float, bnb_usd: float, win_fee_rate: float, gas_usd_roundtrip: float) -> pd.DataFrame:
    df = remaining_fn_df[remaining_fn_df["v21_reason"] == "NEU_WEAK_CONTINUATION_BOTH"].copy()
    if len(df) == 0:
        return pd.DataFrame()
    df["sep_band"] = pd.cut(df["v21_sep"], bins=[-1e-9, 0.02, 0.03, 0.04, 1e9], labels=["0.00-0.02", "0.02-0.03", "0.03-0.04", ">=0.04"])
    df["conf_band"] = pd.cut(df["v21_confidence"], bins=[-1e-9, 0.56, 0.60, 0.64, 1e9], labels=["<0.56", "0.56-0.60", "0.60-0.64", ">0.64"])
    gas_bnb = gas_usd_roundtrip / bnb_usd
    per_win_bnb = (stake_bnb * (1.0 - win_fee_rate)) - gas_bnb
    per_loss_bnb = -stake_bnb - gas_bnb
    rows = []
    grouped = df.groupby(["sep_band", "conf_band"], dropna=False, observed=False)
    for (sep_band, conf_band), sub in grouped:
        if len(sub) == 0:
            continue
        stronger_bull = sub["v21_bull_raw"] >= sub["v21_bear_raw"]
        forced_side = stronger_bull.map(lambda x: "Bull" if bool(x) else "Bear")
        wins = int((forced_side == sub[actual_col]).sum())
        losses = int(len(sub) - wins)
        net_bnb = wins * per_win_bnb + losses * per_loss_bnb
        rows.append({
            "reason": "NEU_WEAK_CONTINUATION_BOTH",
            "sep_band": str(sep_band),
            "conf_band": str(conf_band),
            "rows": int(len(sub)),
            "forced_wins": wins,
            "forced_losses": losses,
            "forced_win_rate": float(wins / len(sub)) if len(sub) else None,
            "forced_net_bnb": float(net_bnb),
            "forced_net_usd": float(net_bnb * bnb_usd),
            "avg_sep": float(sub["v21_sep"].mean()),
            "avg_confidence": float(sub["v21_confidence"].mean()),
            "avg_abs_price_diff": float(sub["actual_price_diff"].abs().mean()) if "actual_price_diff" in sub.columns else None,
        })
    return pd.DataFrame(rows).sort_values(["forced_net_bnb", "forced_win_rate", "rows"], ascending=[False, False, False])


def _append_leaderboard(out_dir: Path, row: dict[str, Any]) -> Path:
    path = out_dir / "leaderboard_v21_profit_first.csv"
    new_df = pd.DataFrame([row])
    if path.exists():
        old_df = pd.read_csv(path)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df
    dedupe_cols = [
        "variant",
        "rollback_weak_conf_low",
        "rollback_weak_conf_high",
        "rollback_weak_low_sep_max",
        "rollback_weak_high_sep_min",
        "stake_bnb",
        "bnb_usd",
        "win_fee_rate",
        "gas_usd_roundtrip",
        "input",
    ]
    keep_cols = [c for c in dedupe_cols if c in df.columns]
    if keep_cols:
        df = df.drop_duplicates(subset=keep_cols, keep="last")
    sort_cols = [c for c in ["net_bnb", "called_only_accuracy", "bets"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    df.to_csv(path, index=False)
    return path


def _write_summary(out_dir: Path, summary: dict[str, Any], variant: str) -> None:
    json_path = out_dir / f"summary_{variant}.json"
    txt_path = out_dir / f"summary_{variant}.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"{variant} profit-first summary", "=" * 60]
    for k, v in summary.items():
        lines.append(f"{k}: {v}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = get_v21_config()
    research_cfg = cfg.get("research", {})
    default_input = research_cfg.get("default_input")
    default_output_dir = research_cfg.get("default_output_dir")

    ap = argparse.ArgumentParser(description="Run v21.3 profit-first rollback sweep for NEU_WEAK_CONTINUATION_BOTH.")
    ap.add_argument("--input", default=default_input)
    ap.add_argument("--output-dir", default=default_output_dir)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epoch-col", default="src_meta_epoch")
    ap.add_argument("--actual-col", default=None)
    ap.add_argument("--stake-bnb", type=float, default=STAKE_BNB_DEFAULT)
    ap.add_argument("--bnb-usd", type=float, default=BNB_USD_DEFAULT)
    ap.add_argument("--win-fee-rate", type=float, default=WIN_FEE_RATE_DEFAULT, help="Platform fee rate on winnings, e.g. 0.05 for 5%%.")
    ap.add_argument("--gas-usd-roundtrip", type=float, default=GAS_USD_ROUNDTRIP_DEFAULT)
    ap.add_argument("--conf-low-grid", default=",".join(map(str, CONF_LOW_GRID_DEFAULT)))
    ap.add_argument("--conf-high-grid", default=",".join(map(str, CONF_HIGH_GRID_DEFAULT)))
    ap.add_argument("--low-sep-max-grid", default=",".join(map(str, LOW_SEP_MAX_GRID_DEFAULT)))
    ap.add_argument("--high-sep-min-grid", default=",".join(map(str, HIGH_SEP_MIN_GRID_DEFAULT)))
    args = ap.parse_args()

    def parse_grid(s: str) -> list[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    conf_low_grid = parse_grid(args.conf_low_grid)
    conf_high_grid = parse_grid(args.conf_high_grid)
    low_sep_grid = parse_grid(args.low_sep_max_grid)
    high_sep_grid = parse_grid(args.high_sep_min_grid)

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _banner("trend_method_v2_1 v21.3 Profit-First Sweep Starting")
    _log(f"[config] input      = {input_path}")
    _log(f"[config] output_dir = {out_dir}")
    _log(f"[config] limit      = {args.limit}")
    _log(f"[config] epoch_col  = {args.epoch_col}")
    _log(f"[config] actual_col = {args.actual_col}")
    _log(f"[config] conf_low_grid      = {conf_low_grid}")
    _log(f"[config] conf_high_grid     = {conf_high_grid}")
    _log(f"[config] low_sep_max_grid   = {low_sep_grid}")
    _log(f"[config] high_sep_min_grid  = {high_sep_grid}")
    _log("-" * 100)

    t0 = time.perf_counter()
    df = _load_frame(input_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    _log(f"[data] loaded rows={len(df):,} cols={len(df.columns):,}")
    diag = _source_schema_diagnostics(df)
    _log(f"[schema] src_total={diag['src_total']} | family_counts={diag['family_counts']}")
    actual_col = args.actual_col or _guess_actual_col(df)
    _log(f"[schema] guessed_actual_col = {actual_col}")
    if not actual_col or actual_col not in df.columns:
        raise KeyError("No actual trend column found. Pass --actual-col explicitly.")

    _log("[stage] running apply_trend_method_v2_1(...) once for sweep ...")
    base_pred = apply_trend_method_v2_1(df, cfg)
    base_pred = _compute_actual_context(base_pred, actual_col)
    if args.epoch_col in base_pred.columns:
        _log(f"[schema] epoch range = {base_pred[args.epoch_col].min()} -> {base_pred[args.epoch_col].max()}")

    combos = []
    for cl, ch, ls, hs in itertools.product(conf_low_grid, conf_high_grid, low_sep_grid, high_sep_grid):
        if cl > ch:
            continue
        if ls >= hs:
            continue
        combos.append((cl, ch, ls, hs))
    _log(f"[sweep] combinations = {len(combos):,}")

    rows = []
    best_row = None
    best_metric = None
    for idx, (cl, ch, ls, hs) in enumerate(combos, start=1):
        rolled_df, rollback_rows = _apply_selective_rollback(base_pred, cl, ch, ls, hs)
        eval_df = rolled_df[rolled_df[actual_col].notna()].copy()
        pred_stats = _pred_stats(eval_df, actual_col)
        wager = _compute_wager_stats(eval_df, actual_col, args.stake_bnb, args.bnb_usd, args.win_fee_rate, args.gas_usd_roundtrip)
        remaining_fn = eval_df[(eval_df["v21_trend"] == "Neutral") & eval_df[actual_col].isin(["Bull", "Bear"])].copy()
        forced_remaining = _forced_remaining_false_neutral(remaining_fn, actual_col, args.stake_bnb, args.bnb_usd, args.win_fee_rate, args.gas_usd_roundtrip)
        row = {
            "variant": "v21_3_sweep",
            "rollback_weak_conf_low": cl,
            "rollback_weak_conf_high": ch,
            "rollback_weak_low_sep_max": ls,
            "rollback_weak_high_sep_min": hs,
            "rows": int(len(eval_df)),
            "rollback_rows": int(len(rollback_rows)),
            "called_only_accuracy": pred_stats["called_accuracy"],
            "bets": wager["bets"],
            "wins": wager["wins"],
            "losses": wager["losses"],
            "net_bnb": wager["net_bnb"],
            "net_usd": wager["net_usd"],
            "pred_bull_acc": pred_stats["pred_bull_acc"],
            "pred_bear_acc": pred_stats["pred_bear_acc"],
            "pred_bull_run_acc": pred_stats["pred_bull_run_acc"],
            "pred_bull_flip_acc": pred_stats["pred_bull_flip_acc"],
            "pred_bear_run_acc": pred_stats["pred_bear_run_acc"],
            "pred_bear_flip_acc": pred_stats["pred_bear_flip_acc"],
            "remaining_false_neutral_rows": forced_remaining.get("rows", 0),
            "remaining_false_neutral_forced_win_rate": forced_remaining.get("win_rate"),
            "remaining_false_neutral_forced_net_bnb": forced_remaining.get("net_bnb"),
            "input": str(input_path),
            "stake_bnb": args.stake_bnb,
            "bnb_usd": args.bnb_usd,
            "win_fee_rate": args.win_fee_rate,
            "gas_usd_roundtrip": args.gas_usd_roundtrip,
        }
        rows.append(row)
        metric = (row["net_bnb"], row["bets"], row["called_only_accuracy"] or -1)
        if best_metric is None or metric > best_metric:
            best_metric = metric
            best_row = row
        if idx % 10 == 0 or idx == len(combos):
            _log(f"[progress] {idx}/{len(combos)} | best_net_bnb={best_row['net_bnb']:.6f} | best_bets={best_row['bets']} | best_called_acc={best_row['called_only_accuracy']:.6f}")

    results_df = pd.DataFrame(rows).sort_values(["net_bnb", "bets", "called_only_accuracy"], ascending=[False, False, False])
    results_path = out_dir / "v21_3_sweep_results.csv"
    results_df.to_csv(results_path, index=False)

    # write top-N summaries and leaderboard append for best row
    top10 = results_df.head(10).copy()
    top10_path = out_dir / "v21_3_sweep_top10.csv"
    top10.to_csv(top10_path, index=False)

    # materialize best run artifacts
    assert best_row is not None
    cl = float(best_row["rollback_weak_conf_low"])
    ch = float(best_row["rollback_weak_conf_high"])
    ls = float(best_row["rollback_weak_low_sep_max"])
    hs = float(best_row["rollback_weak_high_sep_min"])
    best_df, rollback_rows = _apply_selective_rollback(base_pred, cl, ch, ls, hs)
    best_eval = best_df[best_df[actual_col].notna()].copy()
    pred_stats = _pred_stats(best_eval, actual_col)
    wager = _compute_wager_stats(best_eval, actual_col, args.stake_bnb, args.bnb_usd, args.win_fee_rate, args.gas_usd_roundtrip)
    remaining_fn = best_eval[(best_eval["v21_trend"] == "Neutral") & best_eval[actual_col].isin(["Bull", "Bear"])].copy()
    reason_lb = _reason_leaderboard(remaining_fn, actual_col, args.stake_bnb, args.bnb_usd, args.win_fee_rate, args.gas_usd_roundtrip)
    weak_bands = _weak_bands(remaining_fn, actual_col, args.stake_bnb, args.bnb_usd, args.win_fee_rate, args.gas_usd_roundtrip)
    forced_remaining = _forced_remaining_false_neutral(remaining_fn, actual_col, args.stake_bnb, args.bnb_usd, args.win_fee_rate, args.gas_usd_roundtrip)

    pred_path = out_dir / "predictions_v21_3_best.csv"
    best_df.to_csv(pred_path, index=False)
    reason_path = out_dir / "false_neutral_reason_leaderboard_v21_3_best.csv"
    reason_lb.to_csv(reason_path, index=False)
    weak_path = out_dir / "weak_cont_bands_v21_3_best.csv"
    weak_bands.to_csv(weak_path, index=False)

    summary = {
        **best_row,
        **pred_stats,
        "wager": wager,
        "rollback_side_counts": rollback_rows["v21_trend"].value_counts(dropna=False).to_dict() if len(rollback_rows) else {},
        "remaining_false_neutral": forced_remaining,
        "reason_leaderboard_top": reason_lb.head(10).to_dict(orient="records"),
        "weak_bands_top": weak_bands.head(10).to_dict(orient="records"),
        "results_csv": str(results_path),
        "top10_csv": str(top10_path),
        "predictions": str(pred_path),
        "reason_leaderboard_csv": str(reason_path),
        "weak_bands_csv": str(weak_path),
        "actual_col": actual_col,
    }
    _write_summary(out_dir, summary, "v21_3_best")
    leaderboard_path = _append_leaderboard(out_dir, best_row)

    _log(f"[save] sweep_results = {results_path}")
    _log(f"[save] sweep_top10   = {top10_path}")
    _log(f"[save] best_pred     = {pred_path}")
    _log(f"[save] reason_lb     = {reason_path}")
    _log(f"[save] weak_bands    = {weak_path}")
    _log(f"[save] leaderboard   = {leaderboard_path}")
    _log(f"[best] conf_low={cl} conf_high={ch} low_sep_max={ls} high_sep_min={hs}")
    _log(f"[best] net_bnb={best_row['net_bnb']:.6f} bets={best_row['bets']} called_only_accuracy={best_row['called_only_accuracy']:.6f}")
    _banner("trend_method_v2_1 v21.3 Profit-First Sweep Finished")
    _log(f"[done] total_time = {(time.perf_counter() - t0):.2f}s")


if __name__ == "__main__":
    main()
