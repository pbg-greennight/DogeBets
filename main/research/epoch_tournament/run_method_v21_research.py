from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from method_v21 import apply_trend_method_v2_1, get_v21_config


STAKE_BNB_DEFAULT = 0.015
BNB_USD_DEFAULT = 685.0
WIN_FEE_RATE_DEFAULT = 0.05
GAS_USD_ROUNDTRIP_DEFAULT = 0.085
LEADERBOARD_NAME_DEFAULT = "leaderboard_v21_profit_first.csv"


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
        if not part:
            continue
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



def _prepare_eval_df(df: pd.DataFrame, actual_col: str | None, pred_col: str = "v21_trend") -> tuple[pd.DataFrame, str | None]:
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



def _compute_accuracy_block(eval_df: pd.DataFrame) -> dict:
    block: dict = {}
    if eval_df.empty:
        return block

    cmp = eval_df[["__pred_norm", "__actual_norm"]].copy()
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
        sub = cmp[cmp["__pred_norm"] == trend]
        correct = int((sub["__pred_norm"] == sub["__actual_norm"]).sum())
        wrong = int(len(sub) - correct)
        per_pred[trend] = {
            "rows": int(len(sub)),
            "correct": correct,
            "wrong": wrong,
            "accuracy": float(correct / len(sub)) if len(sub) else None,
        }
    block["by_predicted_trend"] = per_pred

    called = cmp[cmp["__pred_norm"] != "Neutral"]
    block["called_only"] = {
        "rows": int(len(called)),
        "correct": int((called["__pred_norm"] == called["__actual_norm"]).sum()),
        "wrong": int((called["__pred_norm"] != called["__actual_norm"]).sum()),
        "accuracy": float((called["__pred_norm"] == called["__actual_norm"]).mean()) if len(called) else None,
    }
    return block



def _compute_wager_block(
    eval_df: pd.DataFrame,
    stake_bnb: float = STAKE_BNB_DEFAULT,
    bnb_usd: float = BNB_USD_DEFAULT,
    win_fee_rate: float = WIN_FEE_RATE_DEFAULT,
    gas_usd_roundtrip: float = GAS_USD_ROUNDTRIP_DEFAULT,
    bet_on_neutral: bool = False,
) -> dict:
    if eval_df.empty:
        return {}

    allowed = {"Bull", "Bear", "Neutral"} if bet_on_neutral else {"Bull", "Bear"}
    bets = eval_df[eval_df["__pred_norm"].isin(allowed)].copy()
    if bets.empty:
        return {}

    gas_bnb_roundtrip = float(gas_usd_roundtrip / bnb_usd) if bnb_usd else 0.0
    win_profit_bnb_before_gas = float(stake_bnb * (1.0 - win_fee_rate))
    lose_profit_bnb_before_gas = float(-stake_bnb)
    win_net_bnb = win_profit_bnb_before_gas - gas_bnb_roundtrip
    lose_net_bnb = lose_profit_bnb_before_gas - gas_bnb_roundtrip

    bets["is_win"] = bets["__pred_norm"] == bets["__actual_norm"]
    bets["bet_pnl_bnb"] = bets["is_win"].map(lambda x: win_net_bnb if x else lose_net_bnb)
    bets["bet_pnl_usd"] = bets["bet_pnl_bnb"] * bnb_usd

    wins = int(bets["is_win"].sum())
    losses = int(len(bets) - wins)

    by_trend = {}
    for trend in sorted(allowed):
        sub = bets[bets["__pred_norm"] == trend].copy()
        tw = int(sub["is_win"].sum())
        tl = int(len(sub) - tw)
        by_trend[trend] = {
            "bets": int(len(sub)),
            "wins": tw,
            "losses": tl,
            "win_rate": float(tw / len(sub)) if len(sub) else None,
            "net_bnb": float(sub["bet_pnl_bnb"].sum()) if len(sub) else 0.0,
            "net_usd": float(sub["bet_pnl_usd"].sum()) if len(sub) else 0.0,
        }

    return {
        "stake_bnb": float(stake_bnb),
        "bnb_usd": float(bnb_usd),
        "stake_usd": float(stake_bnb * bnb_usd),
        "win_fee_rate": float(win_fee_rate),
        "gas_usd_roundtrip": float(gas_usd_roundtrip),
        "gas_bnb_roundtrip": float(gas_bnb_roundtrip),
        "win_net_bnb": float(win_net_bnb),
        "lose_net_bnb": float(lose_net_bnb),
        "bet_on_neutral": bool(bet_on_neutral),
        "bets": int(len(bets)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(bets)) if len(bets) else None,
        "net_bnb": float(bets["bet_pnl_bnb"].sum()),
        "net_usd": float(bets["bet_pnl_usd"].sum()),
        "by_predicted_trend": by_trend,
    }



def _choose_forced_side(row: pd.Series) -> str | None:
    candidates = []
    bull_cont = pd.to_numeric(pd.Series([row.get("v21_bull_continuation_score")]), errors="coerce").iloc[0]
    bear_cont = pd.to_numeric(pd.Series([row.get("v21_bear_continuation_score")]), errors="coerce").iloc[0]
    bull_raw = pd.to_numeric(pd.Series([row.get("v21_bull_raw")]), errors="coerce").iloc[0]
    bear_raw = pd.to_numeric(pd.Series([row.get("v21_bear_raw")]), errors="coerce").iloc[0]

    if pd.notna(bull_cont) and pd.notna(bear_cont) and bull_cont != bear_cont:
        return "Bull" if bull_cont > bear_cont else "Bear"
    if pd.notna(bull_raw) and pd.notna(bear_raw) and bull_raw != bear_raw:
        return "Bull" if bull_raw > bear_raw else "Bear"
    sep = pd.to_numeric(pd.Series([row.get("v21_sep")]), errors="coerce").iloc[0]
    if pd.notna(sep) and pd.notna(bull_raw) and pd.notna(bear_raw) and sep != 0:
        return "Bull" if bull_raw >= bear_raw else "Bear"
    return None



def _compute_trade_pnl(is_win: bool, stake_bnb: float, bnb_usd: float, win_fee_rate: float, gas_usd_roundtrip: float) -> tuple[float, float]:
    gas_bnb_roundtrip = float(gas_usd_roundtrip / bnb_usd) if bnb_usd else 0.0
    if is_win:
        pnl_bnb = stake_bnb * (1.0 - win_fee_rate) - gas_bnb_roundtrip
    else:
        pnl_bnb = -stake_bnb - gas_bnb_roundtrip
    return float(pnl_bnb), float(pnl_bnb * bnb_usd)



def _build_false_neutral_df(eval_df: pd.DataFrame, epoch_col: str) -> pd.DataFrame:
    if eval_df.empty:
        return eval_df.iloc[0:0].copy()
    mask = (eval_df["__pred_norm"] == "Neutral") & (eval_df["__actual_norm"].isin(["Bull", "Bear"]))
    out = eval_df[mask].copy()
    if out.empty:
        return out

    out["forced_side"] = out.apply(_choose_forced_side, axis=1)
    out["forced_side_is_win"] = out["forced_side"] == out["__actual_norm"]

    preferred = [
        epoch_col,
        "__actual_norm",
        "__pred_norm",
        "forced_side",
        "forced_side_is_win",
        "v21_reason",
        "v21_confidence",
        "v21_bull_raw",
        "v21_bear_raw",
        "v21_sep",
        "v21_bull_continuation_score",
        "v21_bear_continuation_score",
        "actual_trend",
        "src_meta_actual_trend",
        "actual_price_diff",
        "src_meta_actual_price_diff",
        "src_meta_price_diff",
        "src_meta_btc_close",
        "src_meta_start_price",
        "src_meta_end_price",
    ]
    cols = [c for c in preferred if c in out.columns]
    remaining = [c for c in out.columns if c not in cols and not c.startswith("__")]
    out = out[cols + remaining]
    rename_map = {"__actual_norm": "actual_trend_norm", "__pred_norm": "predicted_trend_norm"}
    return out.rename(columns=rename_map)



def _compute_false_neutral_summary(false_neutral_df: pd.DataFrame, stake_bnb: float, bnb_usd: float, win_fee_rate: float, gas_usd_roundtrip: float) -> dict:
    if false_neutral_df.empty:
        return {"rows": 0, "reason_counts": {}, "actual_counts": {}, "forced_side": {}, "by_reason": []}

    forced = false_neutral_df[false_neutral_df["forced_side"].isin(["Bull", "Bear"])].copy()
    if not forced.empty:
        pnl = forced["forced_side_is_win"].map(lambda x: _compute_trade_pnl(bool(x), stake_bnb, bnb_usd, win_fee_rate, gas_usd_roundtrip))
        forced["forced_pnl_bnb"] = pnl.map(lambda t: t[0])
        forced["forced_pnl_usd"] = pnl.map(lambda t: t[1])
    else:
        forced["forced_pnl_bnb"] = []
        forced["forced_pnl_usd"] = []

    by_reason_rows = []
    if not forced.empty and "v21_reason" in forced.columns:
        for reason, grp in forced.groupby("v21_reason", dropna=False):
            wins = int(grp["forced_side_is_win"].sum())
            rows = int(len(grp))
            by_reason_rows.append({
                "reason": reason,
                "rows": rows,
                "forced_side_rows": rows,
                "forced_side_wins": wins,
                "forced_side_losses": int(rows - wins),
                "forced_side_win_rate": float(wins / rows) if rows else None,
                "forced_side_net_bnb": float(grp["forced_pnl_bnb"].sum()),
                "forced_side_net_usd": float(grp["forced_pnl_usd"].sum()),
                "avg_confidence": float(pd.to_numeric(grp.get("v21_confidence"), errors="coerce").mean()) if "v21_confidence" in grp.columns else None,
                "avg_sep": float(pd.to_numeric(grp.get("v21_sep"), errors="coerce").mean()) if "v21_sep" in grp.columns else None,
                "avg_abs_price_diff": float(pd.to_numeric(grp.get("actual_price_diff", grp.get("src_meta_actual_price_diff", grp.get("src_meta_price_diff"))), errors="coerce").abs().mean()) if any(c in grp.columns for c in ["actual_price_diff", "src_meta_actual_price_diff", "src_meta_price_diff"]) else None,
            })
        by_reason_rows.sort(key=lambda x: (x["forced_side_net_bnb"], x["forced_side_win_rate"] if x["forced_side_win_rate"] is not None else -1), reverse=True)

    forced_side_summary = {}
    if not forced.empty:
        wins = int(forced["forced_side_is_win"].sum())
        rows = int(len(forced))
        forced_side_summary = {
            "rows": rows,
            "wins": wins,
            "losses": int(rows - wins),
            "win_rate": float(wins / rows) if rows else None,
            "net_bnb": float(forced["forced_pnl_bnb"].sum()),
            "net_usd": float(forced["forced_pnl_usd"].sum()),
            "forced_side_counts": forced["forced_side"].value_counts(dropna=False).to_dict(),
        }

    return {
        "rows": int(len(false_neutral_df)),
        "reason_counts": false_neutral_df["v21_reason"].value_counts(dropna=False).head(20).to_dict() if "v21_reason" in false_neutral_df.columns else {},
        "actual_counts": false_neutral_df["actual_trend_norm"].value_counts(dropna=False).to_dict() if "actual_trend_norm" in false_neutral_df.columns else {},
        "forced_side": forced_side_summary,
        "by_reason": by_reason_rows,
    }



def _compute_basic_summary(
    df: pd.DataFrame,
    epoch_col: str,
    actual_col: str | None,
    stake_bnb: float,
    bnb_usd: float,
    win_fee_rate: float,
    gas_usd_roundtrip: float,
    bet_on_neutral: bool,
) -> tuple[dict, pd.DataFrame]:
    summary: dict = {
        "rows": int(len(df)),
        "epoch_col": epoch_col,
        "trend_counts": df["v21_trend"].value_counts(dropna=False).to_dict() if "v21_trend" in df.columns else {},
        "reason_counts": df["v21_reason"].value_counts(dropna=False).head(20).to_dict() if "v21_reason" in df.columns else {},
        "confidence_mean": float(df["v21_confidence"].mean()) if "v21_confidence" in df.columns and len(df) else None,
        "coverage_non_neutral": float((df["v21_trend"] != "Neutral").mean()) if "v21_trend" in df.columns and len(df) else None,
    }

    eval_df, effective_actual_col = _prepare_eval_df(df, actual_col, "v21_trend")
    summary["actual_col_requested"] = actual_col
    summary["actual_col_used"] = effective_actual_col

    if effective_actual_col and not eval_df.empty:
        summary["accuracy"] = _compute_accuracy_block(eval_df)
        summary["wager"] = _compute_wager_block(
            eval_df,
            stake_bnb=stake_bnb,
            bnb_usd=bnb_usd,
            win_fee_rate=win_fee_rate,
            gas_usd_roundtrip=gas_usd_roundtrip,
            bet_on_neutral=bet_on_neutral,
        )
        false_neutral_df = _build_false_neutral_df(eval_df, epoch_col)
        summary["false_neutral"] = _compute_false_neutral_summary(
            false_neutral_df,
            stake_bnb=stake_bnb,
            bnb_usd=bnb_usd,
            win_fee_rate=win_fee_rate,
            gas_usd_roundtrip=gas_usd_roundtrip,
        )
        return summary, false_neutral_df

    return summary, df.iloc[0:0].copy()



def _append_human_summary(lines: list[str], summary: dict) -> None:
    acc = summary.get("accuracy") or {}
    total = acc.get("total") or {}
    by_pred = acc.get("by_predicted_trend") or {}
    wager = summary.get("wager") or {}
    false_neutral = summary.get("false_neutral") or {}

    lines.append("")
    lines.append("accuracy_stats")
    lines.append("-" * 40)
    if total:
        lines.append(f"total: rows={total.get('rows')} | correct={total.get('correct')} | wrong={total.get('wrong')} | accuracy={total.get('accuracy')}")
        called = acc.get("called_only") or {}
        if called:
            lines.append(f"called_only: rows={called.get('rows')} | correct={called.get('correct')} | wrong={called.get('wrong')} | accuracy={called.get('accuracy')}")
        for trend in ["Bull", "Bear", "Neutral"]:
            row = by_pred.get(trend) or {}
            lines.append(f"pred_{trend.lower()}: rows={row.get('rows')} | correct={row.get('correct')} | wrong={row.get('wrong')} | accuracy={row.get('accuracy')}")
    else:
        lines.append("No actual trend column found; accuracy stats unavailable.")

    lines.append("")
    lines.append("wager_stats")
    lines.append("-" * 40)
    if wager:
        lines.append(
            f"stake: {wager.get('stake_bnb')} BNB @ ${wager.get('bnb_usd')} = ${wager.get('stake_usd')} per bet | fee_on_wins={wager.get('win_fee_rate')} | gas_usd_roundtrip={wager.get('gas_usd_roundtrip')}"
        )
        lines.append(
            f"net_per_win_bnb={wager.get('win_net_bnb')} | net_per_loss_bnb={wager.get('lose_net_bnb')}"
        )
        lines.append(
            f"overall: bets={wager.get('bets')} | wins={wager.get('wins')} | losses={wager.get('losses')} | win_rate={wager.get('win_rate')} | net_bnb={wager.get('net_bnb')} | net_usd={wager.get('net_usd')}"
        )
        for trend in ["Bull", "Bear", "Neutral"]:
            row = (wager.get("by_predicted_trend") or {}).get(trend)
            if row is not None:
                lines.append(
                    f"wager_{trend.lower()}: bets={row.get('bets')} | wins={row.get('wins')} | losses={row.get('losses')} | win_rate={row.get('win_rate')} | net_bnb={row.get('net_bnb')} | net_usd={row.get('net_usd')}"
                )
    else:
        lines.append("No wager stats available because no actual trend column was found.")

    lines.append("")
    lines.append("false_neutral_stats")
    lines.append("-" * 40)
    lines.append(f"rows={false_neutral.get('rows')}")
    lines.append(f"actual_counts={false_neutral.get('actual_counts')}")
    lines.append(f"reason_counts={false_neutral.get('reason_counts')}")
    fs = false_neutral.get("forced_side") or {}
    if fs:
        lines.append(
            f"forced_side: rows={fs.get('rows')} | wins={fs.get('wins')} | losses={fs.get('losses')} | win_rate={fs.get('win_rate')} | net_bnb={fs.get('net_bnb')} | net_usd={fs.get('net_usd')} | side_counts={fs.get('forced_side_counts')}"
        )
    by_reason = false_neutral.get("by_reason") or []
    if by_reason:
        lines.append("top_reason_leakage:")
        for row in by_reason[:10]:
            lines.append(
                f"  {row.get('reason')}: rows={row.get('rows')} | forced_win_rate={row.get('forced_side_win_rate')} | forced_net_bnb={row.get('forced_side_net_bnb')} | avg_sep={row.get('avg_sep')} | avg_conf={row.get('avg_confidence')}"
            )



def _write_summary(summary: dict, out_dir: Path) -> None:
    (out_dir / "summary_v21.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["trend_method_v2_1 research summary", "=" * 40]
    for k, v in summary.items():
        if k in {"accuracy", "wager", "false_neutral", "leaderboard_row"}:
            continue
        lines.append(f"{k}: {v}")
    _append_human_summary(lines, summary)
    (out_dir / "summary_v21.txt").write_text("\n".join(lines), encoding="utf-8")



def _select_debug_rows(df: pd.DataFrame, epoch_col: str, epochs: Iterable[int]) -> pd.DataFrame:
    epochs = list(epochs)
    if not epochs or epoch_col not in df.columns:
        return df.iloc[0:0].copy()
    return df[df[epoch_col].isin(epochs)].copy()



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



def _build_leaderboard_row(summary: dict, input_path: Path, predictions_path: Path) -> dict:
    acc = summary.get("accuracy") or {}
    called = acc.get("called_only") or {}
    wager = summary.get("wager") or {}
    fn = summary.get("false_neutral") or {}
    reasons = fn.get("reason_counts") or {}
    return {
        "run_ts_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": "method_v21",
        "input": str(input_path),
        "predictions": str(predictions_path),
        "rows": summary.get("rows"),
        "coverage_non_neutral": summary.get("coverage_non_neutral"),
        "called_rows": called.get("rows"),
        "called_only_accuracy": called.get("accuracy"),
        "net_bnb": wager.get("net_bnb"),
        "net_usd": wager.get("net_usd"),
        "bets": wager.get("bets"),
        "wins": wager.get("wins"),
        "losses": wager.get("losses"),
        "bull_accuracy": ((acc.get("by_predicted_trend") or {}).get("Bull") or {}).get("accuracy"),
        "bear_accuracy": ((acc.get("by_predicted_trend") or {}).get("Bear") or {}).get("accuracy"),
        "false_neutral_rows": fn.get("rows"),
        "leak_neu_weak_continuation_both": reasons.get("NEU_WEAK_CONTINUATION_BOTH", 0),
        "leak_neu_bull_invalidated": reasons.get("NEU_BULL_INVALIDATED", 0),
        "leak_neu_bear_invalidated": reasons.get("NEU_BEAR_INVALIDATED", 0),
        "leak_neu_close_call": reasons.get("NEU_CLOSE_CALL", 0),
        "forced_false_neutral_win_rate": (fn.get("forced_side") or {}).get("win_rate"),
        "forced_false_neutral_net_bnb": (fn.get("forced_side") or {}).get("net_bnb"),
    }



def _append_leaderboard(row: dict, leaderboard_path: Path) -> None:
    row_df = pd.DataFrame([row])
    if leaderboard_path.exists():
        try:
            old = pd.read_csv(leaderboard_path)
            combined = pd.concat([old, row_df], ignore_index=True)
        except Exception:
            combined = row_df
    else:
        combined = row_df
    sort_cols = [c for c in ["net_bnb", "called_only_accuracy", "bets", "forced_false_neutral_net_bnb"] if c in combined.columns]
    ascending = [False] * len(sort_cols)
    if sort_cols:
        combined = combined.sort_values(sort_cols, ascending=ascending, kind="stable")
    combined.to_csv(leaderboard_path, index=False)



def main() -> None:
    config = get_v21_config()
    research_cfg = config.get("research", {})
    default_input = research_cfg.get("default_input")
    default_output_dir = research_cfg.get("default_output_dir")
    default_limit = research_cfg.get("default_limit")

    ap = argparse.ArgumentParser(description="Run trend_method_v2_1 research model on archived CSV/parquet data.")
    ap.add_argument("--input", default=default_input, help="Input CSV or parquet containing canonical src_* columns or already-mapped columns.")
    ap.add_argument("--output-dir", default=default_output_dir, help="Directory to save predictions and summary files.")
    ap.add_argument("--limit", type=int, default=default_limit, help="Optional row limit for quick testing.")
    ap.add_argument("--epoch-col", default="src_meta_epoch", help="Epoch column name for filtering/debug output.")
    ap.add_argument("--actual-col", default=None, help="Optional actual trend column for accuracy summary.")
    ap.add_argument("--epochs", default=None, help="Comma-separated epochs to export into debug_epochs_v21.csv.")
    ap.add_argument("--predictions-name", default="predictions_v21.csv", help="Predictions output filename (.csv or .parquet).")
    ap.add_argument("--stake-bnb", type=float, default=STAKE_BNB_DEFAULT, help="Stake size in BNB for wager stats.")
    ap.add_argument("--bnb-usd", type=float, default=BNB_USD_DEFAULT, help="BNB price in USD for wager stats.")
    ap.add_argument("--win-fee-rate", type=float, default=WIN_FEE_RATE_DEFAULT, help="Platform fee rate taken from winnings, e.g. 0.05 for 5%%.")
    ap.add_argument("--gas-usd-roundtrip", type=float, default=GAS_USD_ROUNDTRIP_DEFAULT, help="Estimated total gas in USD for place+claim.")
    ap.add_argument("--bet-on-neutral", action="store_true", help="Include Neutral as a bet in wager stats.")
    ap.add_argument("--leaderboard-name", default=LEADERBOARD_NAME_DEFAULT, help="CSV filename for cumulative profit-first leaderboard.")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _banner("trend_method_v2_1 Research Runner Starting")
    _log(f"[config] input      = {input_path}")
    _log(f"[config] output_dir = {out_dir}")
    _log(f"[config] limit      = {args.limit}")
    _log(f"[config] epoch_col   = {args.epoch_col}")
    _log(f"[config] actual_col  = {args.actual_col}")
    _log(f"[config] stake_bnb   = {args.stake_bnb}")
    _log(f"[config] bnb_usd     = {args.bnb_usd}")
    _log(f"[config] win_fee_rate = {args.win_fee_rate}")
    _log(f"[config] gas_usd_roundtrip = {args.gas_usd_roundtrip}")
    _log(f"[config] bet_neutral = {args.bet_on_neutral}")
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
    guessed_actual = _guess_actual_col(df)
    _log(f"[schema] guessed_actual_col = {guessed_actual}")
    if args.epoch_col in df.columns:
        try:
            _log(f"[schema] epoch range = {df[args.epoch_col].min()} -> {df[args.epoch_col].max()}")
        except Exception:
            pass
    _log("-" * 100)

    _log("[stage] running apply_trend_method_v2_1(...) ...")
    t2 = time.perf_counter()
    pred_df = apply_trend_method_v2_1(df, config)
    t3 = time.perf_counter()
    _log(f"[model] completed rows={len(pred_df):,} in {(t3 - t2):.2f}s")

    trend_counts = pred_df["v21_trend"].value_counts(dropna=False).to_dict() if "v21_trend" in pred_df.columns else {}
    reason_counts = pred_df["v21_reason"].value_counts(dropna=False).head(10).to_dict() if "v21_reason" in pred_df.columns else {}
    _log(f"[model] trend_counts={trend_counts}")
    _log(f"[model] top_reasons={reason_counts}")
    _log("-" * 100)

    predictions_path = out_dir / args.predictions_name
    _log(f"[stage] saving predictions -> {predictions_path}")
    _save_frame(pred_df, predictions_path)

    summary, false_neutral_df = _compute_basic_summary(
        pred_df,
        args.epoch_col,
        args.actual_col,
        stake_bnb=args.stake_bnb,
        bnb_usd=args.bnb_usd,
        win_fee_rate=args.win_fee_rate,
        gas_usd_roundtrip=args.gas_usd_roundtrip,
        bet_on_neutral=args.bet_on_neutral,
    )
    summary["input"] = str(input_path)
    summary["predictions"] = str(predictions_path)

    leaderboard_row = _build_leaderboard_row(summary, input_path, predictions_path)
    summary["leaderboard_row"] = leaderboard_row
    leaderboard_path = out_dir / args.leaderboard_name
    _append_leaderboard(leaderboard_row, leaderboard_path)

    _write_summary(summary, out_dir)
    _log(f"[save] summary_json = {out_dir / 'summary_v21.json'}")
    _log(f"[save] summary_txt  = {out_dir / 'summary_v21.txt'}")
    _log(f"[save] leaderboard  = {leaderboard_path}")

    acc = summary.get("accuracy") or {}
    total = acc.get("total") or {}
    by_pred = acc.get("by_predicted_trend") or {}
    wager = summary.get("wager") or {}
    if total:
        _log(f"[stats] total correct={total.get('correct')} wrong={total.get('wrong')} accuracy={total.get('accuracy')}")
        called = acc.get("called_only") or {}
        if called:
            _log(f"[stats] called_only correct={called.get('correct')} wrong={called.get('wrong')} accuracy={called.get('accuracy')}")
        for trend in ["Bull", "Bear", "Neutral"]:
            row = by_pred.get(trend) or {}
            _log(f"[stats] pred_{trend.lower()} correct={row.get('correct')} wrong={row.get('wrong')} accuracy={row.get('accuracy')}")
        if wager:
            _log(
                f"[wager] bets={wager.get('bets')} wins={wager.get('wins')} losses={wager.get('losses')} win_rate={wager.get('win_rate')} net_bnb={wager.get('net_bnb')} net_usd={wager.get('net_usd')}"
            )
            _log(
                f"[wager] per_win_bnb={wager.get('win_net_bnb')} per_loss_bnb={wager.get('lose_net_bnb')} fee_on_wins={wager.get('win_fee_rate')} gas_usd_roundtrip={wager.get('gas_usd_roundtrip')}"
            )
        fn = summary.get("false_neutral") or {}
        _log(f"[false_neutral] rows={fn.get('rows')} actual_counts={fn.get('actual_counts')} reason_counts={fn.get('reason_counts')}")
        fs = fn.get("forced_side") or {}
        if fs:
            _log(f"[forced_neutral] rows={fs.get('rows')} wins={fs.get('wins')} losses={fs.get('losses')} win_rate={fs.get('win_rate')} net_bnb={fs.get('net_bnb')} net_usd={fs.get('net_usd')} side_counts={fs.get('forced_side_counts')}")
        by_reason = fn.get("by_reason") or []
        if by_reason:
            _log(f"[forced_neutral] top_reason_rows={by_reason[:5]}")
    else:
        _log("[stats] No actual trend column found; correct/wrong, wager stats, and false-neutral diagnostics unavailable.")

    if false_neutral_df is not None and not false_neutral_df.empty:
        false_neutral_path = out_dir / "false_neutral_v21.csv"
        false_neutral_df.to_csv(false_neutral_path, index=False)
        _log(f"[save] false_neutral_rows={len(false_neutral_df)} -> {false_neutral_path}")
    else:
        _log("[save] no false-neutral rows found")

    fn_summary = summary.get("false_neutral") or {}
    by_reason = fn_summary.get("by_reason") or []
    if by_reason:
        by_reason_path = out_dir / "false_neutral_reason_leaderboard_v21.csv"
        pd.DataFrame(by_reason).to_csv(by_reason_path, index=False)
        _log(f"[save] false_neutral_reason_leaderboard -> {by_reason_path}")

    debug_epochs = _parse_epochs(args.epochs)
    debug_df = _select_debug_rows(pred_df, args.epoch_col, debug_epochs)
    if len(debug_df):
        debug_path = out_dir / "debug_epochs_v21.csv"
        debug_df.to_csv(debug_path, index=False)
        _log(f"[save] debug_rows={len(debug_df)} -> {debug_path}")
    else:
        _log("[save] no debug epoch subset requested or no matching epochs found")

    _banner("trend_method_v2_1 Research Runner Finished")
    _log(f"[done] predictions = {predictions_path}")
    _log(f"[done] summary     = {out_dir / 'summary_v21.txt'}")
    _log(f"[done] leaderboard = {leaderboard_path}")
    _log(f"[done] total_time  = {(time.perf_counter() - t0):.2f}s")


if __name__ == "__main__":
    main()
