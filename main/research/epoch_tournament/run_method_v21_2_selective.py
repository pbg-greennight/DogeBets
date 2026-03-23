
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

from method_v21 import apply_trend_method_v2_1, get_v21_config

STAKE_BNB_DEFAULT = 0.015
BNB_USD_DEFAULT = 685.0
WIN_FEE_RATE_DEFAULT = 0.05
GAS_USD_ROUNDTRIP_DEFAULT = 0.085

warnings.filterwarnings(
    "ignore",
    message=r"Downcasting object dtype arrays on \\.fillna, \\.ffill, \\.bfill is deprecated.*",
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


def _parse_epochs(raw: str | None) -> list[int]:
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


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
    return str(val).strip()


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
        "y",
        "target",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for col in df.columns:
        cl = col.lower()
        if any(tok in cl for tok in ["actual", "target", "label", "outcome", "winner", "result", "trend"]):
            vals = df[col].dropna().map(_normalize_trend_value)
            uniq = set(vals.dropna().unique().tolist())
            if uniq and uniq.issubset({"Bull", "Bear", "Neutral"}):
                return col
    return None


def _derive_actual_from_price_diff(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    for col in ["actual_price_diff", "src_meta_actual_price_diff", "price_diff", "src_meta_price_diff"]:
        if col in df.columns:
            out_col = "__derived_actual_trend"
            series = pd.to_numeric(df[col], errors="coerce")
            df[out_col] = series.map(lambda x: "Neutral" if pd.isna(x) or x == 0 else ("Bull" if x > 0 else "Bear"))
            return df, out_col
    return df, None


def _prepare_eval_df(df: pd.DataFrame, actual_col: str | None, pred_col: str) -> tuple[pd.DataFrame, str | None]:
    effective_actual_col = actual_col if actual_col and actual_col in df.columns else _guess_actual_col(df)
    work = df.copy()
    if effective_actual_col is None:
        work, effective_actual_col = _derive_actual_from_price_diff(work)
    if effective_actual_col is None or pred_col not in work.columns:
        return work.iloc[0:0].copy(), effective_actual_col

    eval_df = work.copy()
    eval_df["__pred_norm"] = eval_df[pred_col].map(_normalize_trend_value)
    eval_df["__actual_norm"] = eval_df[effective_actual_col].map(_normalize_trend_value)
    eval_df = eval_df[eval_df["__pred_norm"].notna() & eval_df["__actual_norm"].notna()].copy()
    return eval_df, effective_actual_col



def _add_run_flip_context(eval_df: pd.DataFrame, epoch_col: str | None = None) -> pd.DataFrame:
    if eval_df.empty:
        out = eval_df.copy()
        out["__actual_context"] = None
        out["__pred_context"] = None
        return out

    out = eval_df.copy()
    if epoch_col and epoch_col in out.columns:
        out = out.sort_values(epoch_col).copy()

    actual_non_neutral = out["__actual_norm"].where(out["__actual_norm"].isin(["Bull", "Bear"]))
    prev_actual_non_neutral = actual_non_neutral.shift(1).ffill()
    out["__actual_context"] = None
    out.loc[out["__actual_norm"].isin(["Bull", "Bear"]), "__actual_context"] = out.loc[out["__actual_norm"].isin(["Bull", "Bear"]), "__actual_norm"].eq(prev_actual_non_neutral[out["__actual_norm"].isin(["Bull", "Bear"])]).map(lambda x: "run" if x else "flip")

    pred_non_neutral = out["__pred_norm"].where(out["__pred_norm"].isin(["Bull", "Bear"]))
    prev_pred_non_neutral = pred_non_neutral.shift(1).ffill()
    out["__pred_context"] = None
    out.loc[out["__pred_norm"].isin(["Bull", "Bear"]), "__pred_context"] = out.loc[out["__pred_norm"].isin(["Bull", "Bear"]), "__pred_norm"].eq(prev_pred_non_neutral[out["__pred_norm"].isin(["Bull", "Bear"])]).map(lambda x: "run" if x else "flip")
    return out


def _compute_accuracy_block(eval_df: pd.DataFrame, epoch_col: str | None = None) -> dict:
    block: dict = {}
    if eval_df.empty:
        return block
    cmp = _add_run_flip_context(eval_df, epoch_col=epoch_col)[["__pred_norm", "__actual_norm", "__actual_context", "__pred_context"]].copy()
    total_correct = int((cmp["__pred_norm"] == cmp["__actual_norm"]).sum())
    total_wrong = int(len(cmp) - total_correct)
    block["total"] = {
        "rows": int(len(cmp)),
        "correct": total_correct,
        "wrong": total_wrong,
        "accuracy": float(total_correct / len(cmp)),
    }

    per_pred: dict = {}
    for trend in ["Bull", "Bear", "Neutral"]:
        sub = cmp[cmp["__pred_norm"] == trend].copy()
        correct = int((sub["__pred_norm"] == sub["__actual_norm"]).sum())
        wrong = int(len(sub) - correct)
        row = {
            "rows": int(len(sub)),
            "correct": correct,
            "wrong": wrong,
            "accuracy": float(correct / len(sub)) if len(sub) else None,
        }
        if trend in {"Bull", "Bear"}:
            for ctx in ["run", "flip"]:
                sctx = sub[sub["__actual_context"] == ctx]
                ctx_correct = int((sctx["__pred_norm"] == sctx["__actual_norm"]).sum())
                ctx_wrong = int(len(sctx) - ctx_correct)
                row[f"{ctx}_rows"] = int(len(sctx))
                row[f"{ctx}_correct"] = ctx_correct
                row[f"{ctx}_wrong"] = ctx_wrong
                row[f"{ctx}_accuracy"] = float(ctx_correct / len(sctx)) if len(sctx) else None
        per_pred[trend] = row
    block["by_predicted_trend"] = per_pred

    called = cmp[cmp["__pred_norm"] != "Neutral"]
    block["called_only"] = {
        "rows": int(len(called)),
        "correct": int((called["__pred_norm"] == called["__actual_norm"]).sum()),
        "wrong": int((called["__pred_norm"] != called["__actual_norm"]).sum()),
        "accuracy": float((called["__pred_norm"] == called["__actual_norm"]).mean()) if len(called) else None,
    }
    return block


def _compute_wager_block(eval_df: pd.DataFrame, stake_bnb: float, bnb_usd: float, win_fee_rate: float, gas_usd_roundtrip: float, bet_on_neutral: bool) -> dict:
    if eval_df.empty:
        return {}
    allowed = {"Bull", "Bear", "Neutral"} if bet_on_neutral else {"Bull", "Bear"}
    bets = eval_df[eval_df["__pred_norm"].isin(allowed)].copy()
    if bets.empty:
        return {}

    gas_bnb_roundtrip = float(gas_usd_roundtrip / bnb_usd) if bnb_usd else 0.0
    win_net_bnb = float(stake_bnb * (1.0 - win_fee_rate) - gas_bnb_roundtrip)
    lose_net_bnb = float(-stake_bnb - gas_bnb_roundtrip)

    bets["is_win"] = bets["__pred_norm"] == bets["__actual_norm"]
    bets["bet_pnl_bnb"] = bets["is_win"].map(lambda x: win_net_bnb if x else lose_net_bnb)
    bets["bet_pnl_usd"] = bets["bet_pnl_bnb"] * bnb_usd

    by_trend = {}
    for trend in sorted(allowed):
        sub = bets[bets["__pred_norm"] == trend].copy()
        wins = int(sub["is_win"].sum())
        losses = int(len(sub) - wins)
        by_trend[trend] = {
            "bets": int(len(sub)),
            "wins": wins,
            "losses": losses,
            "win_rate": float(wins / len(sub)) if len(sub) else None,
            "net_bnb": float(sub["bet_pnl_bnb"].sum()) if len(sub) else 0.0,
            "net_usd": float(sub["bet_pnl_usd"].sum()) if len(sub) else 0.0,
        }

    wins = int(bets["is_win"].sum())
    losses = int(len(bets) - wins)
    return {
        "stake_bnb": float(stake_bnb),
        "bnb_usd": float(bnb_usd),
        "stake_usd": float(stake_bnb * bnb_usd),
        "win_fee_rate": float(win_fee_rate),
        "gas_usd_roundtrip": float(gas_usd_roundtrip),
        "gas_bnb_roundtrip": float(gas_bnb_roundtrip),
        "win_net_bnb": win_net_bnb,
        "lose_net_bnb": lose_net_bnb,
        "bet_on_neutral": bool(bet_on_neutral),
        "bets": int(len(bets)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(bets)) if len(bets) else None,
        "net_bnb": float(bets["bet_pnl_bnb"].sum()),
        "net_usd": float(bets["bet_pnl_usd"].sum()),
        "by_predicted_trend": by_trend,
    }


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
    return {"src_total": len(src_cols), "family_counts": {k: len(v) for k, v in fams.items()}}


def _select_debug_rows(df: pd.DataFrame, epoch_col: str, epochs: Iterable[int]) -> pd.DataFrame:
    epochs = list(epochs)
    if not epochs or epoch_col not in df.columns:
        return df.iloc[0:0].copy()
    return df[df[epoch_col].isin(epochs)].copy()


def _stronger_side(row: pd.Series, bull_col: str = "v21_bull_raw", bear_col: str = "v21_bear_raw") -> str:
    bull = pd.to_numeric(row.get(bull_col), errors="coerce")
    bear = pd.to_numeric(row.get(bear_col), errors="coerce")
    bull = -1e18 if pd.isna(bull) else float(bull)
    bear = -1e18 if pd.isna(bear) else float(bear)
    return "Bull" if bull >= bear else "Bear"


def _apply_selective_neutral_rollback(df: pd.DataFrame, conf_low: float, conf_high: float, low_sep_max: float, high_sep_min: float) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    pred_col = "v21r_trend"
    reason_col = "v21r_reason"
    out[pred_col] = out["v21_trend"]
    out[reason_col] = out["v21_reason"]
    out["v21r_rollback_applied"] = False
    out["v21r_rollback_bucket"] = None
    out["v21r_forced_side"] = None

    sep = pd.to_numeric(out.get("v21_sep"), errors="coerce").fillna(-1.0)
    conf = pd.to_numeric(out.get("v21_confidence"), errors="coerce").fillna(-1.0)
    mask = (
        out["v21_trend"].eq("Neutral")
        & out["v21_reason"].eq("NEU_WEAK_CONTINUATION_BOTH")
        & conf.ge(conf_low)
        & conf.lt(conf_high)
        & (sep.lt(low_sep_max) | sep.ge(high_sep_min))
    )
    if mask.any():
        forced = out.loc[mask].apply(_stronger_side, axis=1)
        out.loc[mask, pred_col] = forced.values
        out.loc[mask, reason_col] = out.loc[mask, "v21_reason"].astype(str) + "__ROLLBACK_FORCED"
        out.loc[mask, "v21r_rollback_applied"] = True
        out.loc[mask, "v21r_rollback_bucket"] = "NEU_WEAK_CONTINUATION_BOTH"
        out.loc[mask, "v21r_forced_side"] = forced.values

    meta = {
        "rollback_reason": "NEU_WEAK_CONTINUATION_BOTH",
        "rollback_conf_low": float(conf_low),
        "rollback_conf_high": float(conf_high),
        "rollback_low_sep_max": float(low_sep_max),
        "rollback_high_sep_min": float(high_sep_min),
        "rollback_rows": int(mask.sum()),
        "rollback_side_counts": out.loc[mask, "v21r_forced_side"].value_counts(dropna=False).to_dict() if mask.any() else {},
    }
    return out, meta


def _false_neutral_df(eval_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    if eval_df.empty:
        return eval_df.iloc[0:0].copy()
    f = eval_df[(eval_df[pred_col].eq("Neutral")) & (eval_df["__actual_norm"].isin(["Bull", "Bear"]))].copy()
    if f.empty:
        return f
    f["forced_side"] = f.apply(_stronger_side, axis=1)
    f["forced_side_is_win"] = f["forced_side"] == f["__actual_norm"]
    return f


def _forced_neutral_summary(false_neutral_df: pd.DataFrame, stake_bnb: float, bnb_usd: float, win_fee_rate: float, gas_usd_roundtrip: float) -> dict:
    if false_neutral_df.empty:
        return {}
    gas_bnb_roundtrip = float(gas_usd_roundtrip / bnb_usd) if bnb_usd else 0.0
    win_net_bnb = float(stake_bnb * (1.0 - win_fee_rate) - gas_bnb_roundtrip)
    lose_net_bnb = float(-stake_bnb - gas_bnb_roundtrip)
    df = false_neutral_df.copy()
    df["forced_pnl_bnb"] = df["forced_side_is_win"].map(lambda x: win_net_bnb if x else lose_net_bnb)
    df["forced_pnl_usd"] = df["forced_pnl_bnb"] * bnb_usd
    wins = int(df["forced_side_is_win"].sum())
    losses = int(len(df) - wins)
    return {
        "rows": int(len(df)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(df)) if len(df) else None,
        "net_bnb": float(df["forced_pnl_bnb"].sum()),
        "net_usd": float(df["forced_pnl_usd"].sum()),
        "side_counts": df["forced_side"].value_counts(dropna=False).to_dict(),
    }


def _reason_leaderboard(false_neutral_df: pd.DataFrame, stake_bnb: float, bnb_usd: float, win_fee_rate: float, gas_usd_roundtrip: float) -> pd.DataFrame:
    if false_neutral_df.empty:
        return pd.DataFrame()
    gas_bnb_roundtrip = float(gas_usd_roundtrip / bnb_usd) if bnb_usd else 0.0
    win_net_bnb = float(stake_bnb * (1.0 - win_fee_rate) - gas_bnb_roundtrip)
    lose_net_bnb = float(-stake_bnb - gas_bnb_roundtrip)
    df = false_neutral_df.copy()
    df["forced_pnl_bnb"] = df["forced_side_is_win"].map(lambda x: win_net_bnb if x else lose_net_bnb)
    df["forced_pnl_usd"] = df["forced_pnl_bnb"] * bnb_usd
    rows = []
    for reason, sub in df.groupby("v21_reason", dropna=False):
        wins = int(sub["forced_side_is_win"].sum())
        losses = int(len(sub) - wins)
        rows.append({
            "reason": reason,
            "rows": int(len(sub)),
            "forced_side_rows": int(len(sub)),
            "forced_side_wins": wins,
            "forced_side_losses": losses,
            "forced_side_win_rate": float(wins / len(sub)) if len(sub) else None,
            "forced_side_net_bnb": float(sub["forced_pnl_bnb"].sum()),
            "forced_side_net_usd": float(sub["forced_pnl_usd"].sum()),
            "avg_confidence": float(pd.to_numeric(sub.get("v21_confidence"), errors="coerce").mean()),
            "avg_sep": float(pd.to_numeric(sub.get("v21_sep"), errors="coerce").mean()),
            "avg_abs_price_diff": float(pd.to_numeric(sub.get("actual_price_diff", sub.get("src_meta_actual_price_diff")), errors="coerce").abs().mean()),
        })
    return pd.DataFrame(rows).sort_values(["forced_side_net_bnb", "forced_side_win_rate", "rows"], ascending=[False, False, False])


def _assign_band(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(series, errors="coerce"), bins=bins, labels=labels, include_lowest=True, right=False)


def _weak_cont_bands(false_neutral_df: pd.DataFrame, stake_bnb: float, bnb_usd: float, win_fee_rate: float, gas_usd_roundtrip: float) -> pd.DataFrame:
    if false_neutral_df.empty:
        return pd.DataFrame()
    df = false_neutral_df[false_neutral_df["v21_reason"].eq("NEU_WEAK_CONTINUATION_BOTH")].copy()
    if df.empty:
        return pd.DataFrame()
    gas_bnb_roundtrip = float(gas_usd_roundtrip / bnb_usd) if bnb_usd else 0.0
    win_net_bnb = float(stake_bnb * (1.0 - win_fee_rate) - gas_bnb_roundtrip)
    lose_net_bnb = float(-stake_bnb - gas_bnb_roundtrip)
    df["forced_pnl_bnb"] = df["forced_side_is_win"].map(lambda x: win_net_bnb if x else lose_net_bnb)
    df["forced_pnl_usd"] = df["forced_pnl_bnb"] * bnb_usd
    df["sep_band"] = _assign_band(df.get("v21_sep"), [0.0, 0.02, 0.03, 0.04, 1e9], ["0.00-0.02", "0.02-0.03", "0.03-0.04", ">=0.04"])
    df["conf_band"] = _assign_band(df.get("v21_confidence"), [0.0, 0.56, 0.60, 0.64, 1e9], ["0.00-0.56", "0.56-0.60", "0.60-0.64", ">=0.64"])
    rows = []
    grouped = df.groupby(["sep_band", "conf_band"], dropna=False, observed=False)
    for (sep_band, conf_band), sub in grouped:
        wins = int(sub["forced_side_is_win"].sum())
        losses = int(len(sub) - wins)
        rows.append({
            "reason": "NEU_WEAK_CONTINUATION_BOTH",
            "sep_band": str(sep_band),
            "conf_band": str(conf_band),
            "rows": int(len(sub)),
            "forced_wins": wins,
            "forced_losses": losses,
            "forced_win_rate": float(wins / len(sub)) if len(sub) else None,
            "forced_net_bnb": float(sub["forced_pnl_bnb"].sum()),
            "forced_net_usd": float(sub["forced_pnl_usd"].sum()),
            "avg_sep": float(pd.to_numeric(sub.get("v21_sep"), errors="coerce").mean()),
            "avg_confidence": float(pd.to_numeric(sub.get("v21_confidence"), errors="coerce").mean()),
            "avg_abs_price_diff": float(pd.to_numeric(sub.get("actual_price_diff", sub.get("src_meta_actual_price_diff")), errors="coerce").abs().mean()),
        })
    return pd.DataFrame(rows).sort_values(["forced_net_bnb", "forced_win_rate", "rows"], ascending=[False, False, False])


def _append_leaderboard(out_dir: Path, row: dict) -> Path:
    path = out_dir / "leaderboard_v21_profit_first.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        new = pd.DataFrame([row])
    dedupe_cols = [c for c in ["run_label", "input", "rows", "rollback_weak_conf_low", "rollback_weak_conf_high", "rollback_weak_low_sep_max", "rollback_weak_high_sep_min"] if c in new.columns]
    if dedupe_cols:
        new = new.drop_duplicates(subset=dedupe_cols, keep="last")
    sort_cols = [c for c in ["net_bnb", "called_only_accuracy", "bets"] if c in new.columns]
    if sort_cols:
        asc = [False] * len(sort_cols)
        new = new.sort_values(sort_cols, ascending=asc)
    new.to_csv(path, index=False)
    return path


def _write_summary(summary: dict, out_dir: Path, name: str) -> None:
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [name, "=" * 60]
    for k, v in summary.items():
        lines.append(f"{k}: {v}")
    (out_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = get_v21_config()
    research_cfg = config.get("research", {})
    default_input = research_cfg.get("default_input")
    default_output_dir = research_cfg.get("default_output_dir")
    default_limit = research_cfg.get("default_limit")

    ap = argparse.ArgumentParser(description="Run trend_method_v2_1 profit-first research with selective neutral rollback diagnostics.")
    ap.add_argument("--input", default=default_input)
    ap.add_argument("--output-dir", default=default_output_dir)
    ap.add_argument("--limit", type=int, default=default_limit)
    ap.add_argument("--epoch-col", default="src_meta_epoch")
    ap.add_argument("--actual-col", default=None)
    ap.add_argument("--epochs", default=None)
    ap.add_argument("--predictions-name", default="predictions_v21_2_selective.csv")
    ap.add_argument("--stake-bnb", type=float, default=STAKE_BNB_DEFAULT)
    ap.add_argument("--bnb-usd", type=float, default=BNB_USD_DEFAULT)
    ap.add_argument("--win-fee-rate", type=float, default=WIN_FEE_RATE_DEFAULT, help="Platform fee rate taken from winnings, e.g. 0.05 for 5%%.")
    ap.add_argument("--gas-usd-roundtrip", type=float, default=GAS_USD_ROUNDTRIP_DEFAULT)
    ap.add_argument("--bet-neutral", action="store_true")
    ap.add_argument("--rollback-weak-conf-low", type=float, default=0.60)
    ap.add_argument("--rollback-weak-conf-high", type=float, default=0.64)
    ap.add_argument("--rollback-weak-low-sep-max", type=float, default=0.02)
    ap.add_argument("--rollback-weak-high-sep-min", type=float, default=0.04)
    ap.add_argument("--run-label", default="v21.2_selective_weak_continuation_rollback")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _banner("trend_method_v2_1 Profit-First Research Runner Starting")
    _log(f"[config] input      = {input_path}")
    _log(f"[config] output_dir = {out_dir}")
    _log(f"[config] limit      = {args.limit}")
    _log(f"[config] epoch_col   = {args.epoch_col}")
    _log(f"[config] actual_col  = {args.actual_col}")
    _log(f"[config] stake_bnb   = {args.stake_bnb}")
    _log(f"[config] bnb_usd     = {args.bnb_usd}")
    _log(f"[config] win_fee_rate = {args.win_fee_rate}")
    _log(f"[config] gas_usd_roundtrip = {args.gas_usd_roundtrip}")
    _log(f"[config] bet_neutral = {args.bet_neutral}")
    _log(f"[config] rollback_weak_conf_low = {args.rollback_weak_conf_low}")
    _log(f"[config] rollback_weak_conf_high = {args.rollback_weak_conf_high}")
    _log(f"[config] rollback_weak_low_sep_max = {args.rollback_weak_low_sep_max}")
    _log(f"[config] rollback_weak_high_sep_min = {args.rollback_weak_high_sep_min}")
    _log("-" * 100)

    t0 = time.perf_counter()
    _log("[stage] loading archived dataset...")
    df = _load_frame(input_path)
    t1 = time.perf_counter()
    _log(f"[data] loaded rows={len(df):,} cols={len(df.columns):,} in {(t1 - t0):.2f}s")
    if args.limit is not None:
        df = df.head(args.limit).copy()
        _log(f"[data] applied limit -> rows={len(df):,}")
    diag = _source_schema_diagnostics(df)
    _log(f"[schema] src_total={diag['src_total']} | family_counts={diag['family_counts']}")
    guessed_actual = _guess_actual_col(df) if args.actual_col is None else args.actual_col
    _log(f"[schema] guessed_actual_col = {guessed_actual}")
    if args.epoch_col in df.columns:
        _log(f"[schema] epoch range = {df[args.epoch_col].min()} -> {df[args.epoch_col].max()}")
    _log("-" * 100)

    _log("[stage] running apply_trend_method_v2_1(...) ...")
    t2 = time.perf_counter()
    pred_df = apply_trend_method_v2_1(df, config)
    t3 = time.perf_counter()
    _log(f"[model] completed rows={len(pred_df):,} in {(t3 - t2):.2f}s")
    _log(f"[model] trend_counts={pred_df['v21_trend'].value_counts(dropna=False).to_dict() if 'v21_trend' in pred_df.columns else {}}")
    _log(f"[model] top_reasons={pred_df['v21_reason'].value_counts(dropna=False).head(10).to_dict() if 'v21_reason' in pred_df.columns else {}}")
    _log("-" * 100)

    pred_df, rollback_meta = _apply_selective_neutral_rollback(pred_df, args.rollback_weak_conf_low, args.rollback_weak_conf_high, args.rollback_weak_low_sep_max, args.rollback_weak_high_sep_min)
    _log(f"[rollback] rows={rollback_meta['rollback_rows']} side_counts={rollback_meta['rollback_side_counts']}")

    eval_df, actual_col_used = _prepare_eval_df(pred_df, args.actual_col, pred_col="v21r_trend")
    accuracy = _compute_accuracy_block(eval_df, epoch_col=args.epoch_col)
    wager = _compute_wager_block(eval_df, args.stake_bnb, args.bnb_usd, args.win_fee_rate, args.gas_usd_roundtrip, args.bet_neutral)
    false_neutral = _false_neutral_df(eval_df, pred_col="v21r_trend")
    forced_summary = _forced_neutral_summary(false_neutral, args.stake_bnb, args.bnb_usd, args.win_fee_rate, args.gas_usd_roundtrip)
    reason_lb = _reason_leaderboard(false_neutral, args.stake_bnb, args.bnb_usd, args.win_fee_rate, args.gas_usd_roundtrip)
    weak_bands = _weak_cont_bands(false_neutral, args.stake_bnb, args.bnb_usd, args.win_fee_rate, args.gas_usd_roundtrip)

    predictions_path = out_dir / args.predictions_name
    _log(f"[stage] saving predictions -> {predictions_path}")
    _save_frame(pred_df, predictions_path)

    false_path = out_dir / "false_neutral_v21_2.csv"
    reason_path = out_dir / "false_neutral_reason_leaderboard_v21_2.csv"
    bands_path = out_dir / "weak_cont_bands_v21_2.csv"
    false_neutral.to_csv(false_path, index=False)
    if not reason_lb.empty:
        reason_lb.to_csv(reason_path, index=False)
    if not weak_bands.empty:
        weak_bands.to_csv(bands_path, index=False)

    summary = {
        "run_label": args.run_label,
        "rows": int(len(pred_df)),
        "epoch_col": args.epoch_col,
        "trend_counts": pred_df["v21r_trend"].value_counts(dropna=False).to_dict() if "v21r_trend" in pred_df.columns else {},
        "reason_counts": pred_df["v21r_reason"].value_counts(dropna=False).head(20).to_dict() if "v21r_reason" in pred_df.columns else {},
        "confidence_mean": float(pd.to_numeric(pred_df.get("v21_confidence"), errors="coerce").mean()) if len(pred_df) else None,
        "coverage_non_neutral": float((pred_df["v21r_trend"] != "Neutral").mean()) if "v21r_trend" in pred_df.columns and len(pred_df) else None,
        "actual_col_requested": args.actual_col,
        "actual_col_used": actual_col_used,
        "rollback": rollback_meta,
        "accuracy": accuracy,
        "wager": wager,
        "false_neutral": {
            "rows": int(len(false_neutral)),
            "actual_counts": false_neutral["__actual_norm"].value_counts(dropna=False).to_dict() if len(false_neutral) else {},
            "reason_counts": false_neutral["v21_reason"].value_counts(dropna=False).to_dict() if len(false_neutral) else {},
        },
        "forced_neutral": forced_summary,
        "input": str(input_path),
        "predictions": str(predictions_path),
        "false_neutral_path": str(false_path),
        "false_neutral_reason_leaderboard_path": str(reason_path),
        "weak_cont_bands_path": str(bands_path),
    }
    _write_summary(summary, out_dir, "summary_v21_2_selective")

    leaderboard_row = {
        "run_label": args.run_label,
        "input": str(input_path),
        "rows": int(len(pred_df)),
        "coverage_non_neutral": summary["coverage_non_neutral"],
        "called_only_accuracy": accuracy.get("called_only", {}).get("accuracy"),
        "bets": wager.get("bets"),
        "wins": wager.get("wins"),
        "losses": wager.get("losses"),
        "win_rate": wager.get("win_rate"),
        "net_bnb": wager.get("net_bnb"),
        "net_usd": wager.get("net_usd"),
        "bull_accuracy": accuracy.get("by_predicted_trend", {}).get("Bull", {}).get("accuracy"),
        "bear_accuracy": accuracy.get("by_predicted_trend", {}).get("Bear", {}).get("accuracy"),
        "false_neutral_rows": int(len(false_neutral)),
        "fn_neu_weak_both": int(summary["false_neutral"]["reason_counts"].get("NEU_WEAK_CONTINUATION_BOTH", 0)),
        "fn_neu_close_call": int(summary["false_neutral"]["reason_counts"].get("NEU_CLOSE_CALL", 0)),
        "fn_neu_bull_invalidated": int(summary["false_neutral"]["reason_counts"].get("NEU_BULL_INVALIDATED", 0)),
        "fn_neu_bear_invalidated": int(summary["false_neutral"]["reason_counts"].get("NEU_BEAR_INVALIDATED", 0)),
        "forced_neutral_net_bnb": forced_summary.get("net_bnb"),
        "forced_neutral_win_rate": forced_summary.get("win_rate"),
        "rollback_rows": rollback_meta.get("rollback_rows"),
        "rollback_weak_conf_low": args.rollback_weak_conf_low,
        "rollback_weak_conf_high": args.rollback_weak_conf_high,
        "rollback_weak_low_sep_max": args.rollback_weak_low_sep_max,
        "rollback_weak_high_sep_min": args.rollback_weak_high_sep_min,
    }
    leaderboard_path = _append_leaderboard(out_dir, leaderboard_row)

    _log(f"[save] summary_json = {out_dir / 'summary_v21_2_selective.json'}")
    _log(f"[save] summary_txt  = {out_dir / 'summary_v21_2_selective.txt'}")
    _log(f"[save] leaderboard  = {leaderboard_path}")

    if accuracy:
        _log(f"[stats] total correct={accuracy['total']['correct']} wrong={accuracy['total']['wrong']} accuracy={accuracy['total']['accuracy']}")
        called = accuracy.get('called_only', {})
        _log(f"[stats] called_only correct={called.get('correct')} wrong={called.get('wrong')} accuracy={called.get('accuracy')}")
        for trend in ["Bull", "Bear", "Neutral"]:
            d = accuracy.get("by_predicted_trend", {}).get(trend, {})
            _log(f"[stats] pred_{trend.lower()} correct={d.get('correct')} wrong={d.get('wrong')} accuracy={d.get('accuracy')}")
    if wager:
        _log(f"[wager] bets={wager['bets']} wins={wager['wins']} losses={wager['losses']} win_rate={wager['win_rate']} net_bnb={wager['net_bnb']} net_usd={wager['net_usd']}")
        _log(f"[wager] per_win_bnb={wager['win_net_bnb']} per_loss_bnb={wager['lose_net_bnb']} fee_on_wins={wager['win_fee_rate']} gas_usd_roundtrip={wager['gas_usd_roundtrip']}")
    _log(f"[false_neutral] rows={len(false_neutral)} actual_counts={summary['false_neutral']['actual_counts']} reason_counts={summary['false_neutral']['reason_counts']}")
    if forced_summary:
        _log(f"[forced_neutral] rows={forced_summary['rows']} wins={forced_summary['wins']} losses={forced_summary['losses']} win_rate={forced_summary['win_rate']} net_bnb={forced_summary['net_bnb']} net_usd={forced_summary['net_usd']} side_counts={forced_summary['side_counts']}")
    if not reason_lb.empty:
        _log(f"[forced_neutral] top_reason_rows={reason_lb.head(4).to_dict(orient='records')}")
    if not weak_bands.empty:
        _log(f"[weak_bands] top_rows={weak_bands.head(8).to_dict(orient='records')}")

    debug_epochs = _parse_epochs(args.epochs)
    debug_df = _select_debug_rows(pred_df, args.epoch_col, debug_epochs)
    if len(debug_df):
        debug_path = out_dir / "debug_epochs_v21_2_selective.csv"
        debug_df.to_csv(debug_path, index=False)
        _log(f"[save] debug_rows={len(debug_df)} -> {debug_path}")
    else:
        _log("[save] no debug epoch subset requested or no matching epochs found")

    _banner("trend_method_v2_1 Profit-First Research Runner Finished")
    _log(f"[done] predictions = {predictions_path}")
    _log(f"[done] summary     = {out_dir / 'summary_v21_2_selective.txt'}")
    _log(f"[done] leaderboard = {leaderboard_path}")
    _log(f"[done] total_time  = {(time.perf_counter() - t0):.2f}s")


if __name__ == "__main__":
    main()
