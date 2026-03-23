from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
)


def compute_prediction_metrics(pred_df: pd.DataFrame, wager_cfg: dict | None = None) -> dict:
    if pred_df.empty:
        return {}

    wager_cfg = wager_cfg or {}

    stake_bnb = float(wager_cfg.get("stake_bnb", 0.015))
    bnb_usd_price = float(wager_cfg.get("bnb_usd_price", 685.0))
    payout_win_multiple = float(wager_cfg.get("payout_win_multiple", 0.95))
    site_fee_on_win_rate = float(wager_cfg.get("site_fee_on_win_rate", 0.05))
    gas_bet_usd = float(wager_cfg.get("gas_bet_usd", 0.85))
    gas_claim_usd = float(wager_cfg.get("gas_claim_usd", 0.85))

    gas_bet_bnb = gas_bet_usd / bnb_usd_price
    gas_claim_bnb = gas_claim_usd / bnb_usd_price

    out: dict = {}

    out["epochs_seen"] = int(len(pred_df))
    out["calls_made"] = int(pred_df["wager"].sum())
    out["coverage_pct"] = float(pred_df["wager"].mean() * 100.0)
    out["skip_rows"] = int((pred_df["pred_label"] == -1).sum())

    # Directional rows only = Bull/Bear predictions, not Skip
    directional = pred_df[pred_df["pred_label"].isin([0, 1])].copy()
    out["directional_rows"] = int(len(directional))
    out["directional_coverage_pct"] = float((len(directional) / len(pred_df)) * 100.0) if len(pred_df) > 0 else 0.0

    if directional.empty:
        out["overall_accuracy"] = float("nan")
        out["avg_confidence"] = float(pred_df["confidence"].mean()) if "confidence" in pred_df.columns else float("nan")
        out["avg_prob_bull"] = float(pred_df["prob_bull"].mean()) if "prob_bull" in pred_df.columns else float("nan")
        out["avg_prob_bear"] = float(pred_df["prob_bear"].mean()) if "prob_bear" in pred_df.columns else float("nan")

        out["log_loss"] = float("nan")
        out["brier_score"] = float("nan")

        out["bull_precision"] = float("nan")
        out["bull_recall"] = float("nan")
        out["bear_precision"] = float("nan")
        out["bear_recall"] = float("nan")

        out["tn_bear_ok"] = 0
        out["fp_false_bull"] = 0
        out["fn_false_bear"] = 0
        out["tp_bull_ok"] = 0
    else:
        all_true = directional["actual_label"].astype(int)
        all_pred = directional["pred_label"].astype(int)
        all_prob_bull = directional["prob_bull"].clip(1e-9, 1 - 1e-9)

        out["overall_accuracy"] = float(accuracy_score(all_true, all_pred))
        out["avg_confidence"] = float(pred_df["confidence"].mean())
        out["avg_prob_bull"] = float(pred_df["prob_bull"].mean())
        out["avg_prob_bear"] = float(pred_df["prob_bear"].mean())

        out["log_loss"] = float(log_loss(all_true, all_prob_bull, labels=[0, 1]))
        out["brier_score"] = float(np.mean((all_prob_bull - all_true) ** 2))

        out["bull_precision"] = float(precision_score(all_true, all_pred, pos_label=1, zero_division=0))
        out["bull_recall"] = float(recall_score(all_true, all_pred, pos_label=1, zero_division=0))
        out["bear_precision"] = float(precision_score(all_true, all_pred, pos_label=0, zero_division=0))
        out["bear_recall"] = float(recall_score(all_true, all_pred, pos_label=0, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(all_true, all_pred, labels=[0, 1]).ravel()
        out["tn_bear_ok"] = int(tn)
        out["fp_false_bull"] = int(fp)
        out["fn_false_bear"] = int(fn)
        out["tp_bull_ok"] = int(tp)

    out["stake_bnb"] = stake_bnb
    out["bnb_usd_price"] = bnb_usd_price
    out["payout_win_multiple"] = payout_win_multiple
    out["site_fee_on_win_rate"] = site_fee_on_win_rate
    out["gas_bet_usd"] = gas_bet_usd
    out["gas_claim_usd"] = gas_claim_usd

    # Called rows only: wager=True AND actual directional call
    called = pred_df[(pred_df["wager"] == True) & (pred_df["pred_label"].isin([0, 1]))].copy()
    if called.empty:
        return _finalize_empty_called_metrics(out)

    called["is_hit"] = (called["actual_label"].astype(int) == called["pred_label"].astype(int))

    gross_win_profit_bnb = stake_bnb * payout_win_multiple
    site_fee_bnb = gross_win_profit_bnb * site_fee_on_win_rate
    net_win_profit_bnb = gross_win_profit_bnb - site_fee_bnb

    called["pnl_bnb"] = np.where(
        called["is_hit"],
        net_win_profit_bnb - gas_bet_bnb - gas_claim_bnb,
        -stake_bnb - gas_bet_bnb
    )
    called["pnl_usd"] = called["pnl_bnb"] * bnb_usd_price

    called["pnl_units_raw"] = np.where(
        called["is_hit"],
        payout_win_multiple * (1.0 - site_fee_on_win_rate),
        -1.0
    )
    called["pnl_units_after_cost"] = called["pnl_bnb"] / stake_bnb

    out["called_accuracy"] = float(called["is_hit"].mean())

    bull_called = called[called["pred_label"] == 1].copy()
    bear_called = called[called["pred_label"] == 0].copy()

    out["bull_calls"] = int(len(bull_called))
    out["bear_calls"] = int(len(bear_called))
    out["coverage_bull_pct"] = float((len(bull_called) / len(pred_df)) * 100.0)
    out["coverage_bear_pct"] = float((len(bear_called) / len(pred_df)) * 100.0)

    out["called_bull_accuracy"] = float(bull_called["is_hit"].mean()) if not bull_called.empty else float("nan")
    out["called_bear_accuracy"] = float(bear_called["is_hit"].mean()) if not bear_called.empty else float("nan")

    out["bull_net_units"] = float(bull_called["pnl_units_after_cost"].sum()) if not bull_called.empty else 0.0
    out["bear_net_units"] = float(bear_called["pnl_units_after_cost"].sum()) if not bear_called.empty else 0.0

    out["bull_net_bnb"] = float(bull_called["pnl_bnb"].sum()) if not bull_called.empty else 0.0
    out["bear_net_bnb"] = float(bear_called["pnl_bnb"].sum()) if not bear_called.empty else 0.0

    out["bull_net_usd"] = float(bull_called["pnl_usd"].sum()) if not bull_called.empty else 0.0
    out["bear_net_usd"] = float(bear_called["pnl_usd"].sum()) if not bear_called.empty else 0.0

    out["gross_wins"] = int(called["is_hit"].sum())
    out["gross_losses"] = int((~called["is_hit"]).sum())

    out["net_units"] = float(called["pnl_units_after_cost"].sum())
    out["net_bnb"] = float(called["pnl_bnb"].sum())
    out["net_usd"] = float(called["pnl_usd"].sum())

    out["avg_bnb_per_bet"] = float(called["pnl_bnb"].mean())
    out["avg_usd_per_bet"] = float(called["pnl_usd"].mean())

    wins = called[called["is_hit"] == True].copy()
    losses = called[called["is_hit"] == False].copy()

    out["avg_conf_win"] = float(wins["confidence"].mean()) if not wins.empty else float("nan")
    out["avg_conf_loss"] = float(losses["confidence"].mean()) if not losses.empty else float("nan")

    out["bull_avg_conf"] = float(bull_called["confidence"].mean()) if not bull_called.empty else float("nan")
    out["bear_avg_conf"] = float(bear_called["confidence"].mean()) if not bear_called.empty else float("nan")

    bull_wins = bull_called[bull_called["is_hit"] == True]
    bull_losses = bull_called[bull_called["is_hit"] == False]
    bear_wins = bear_called[bear_called["is_hit"] == True]
    bear_losses = bear_called[bear_called["is_hit"] == False]

    out["bull_avg_conf_win"] = float(bull_wins["confidence"].mean()) if not bull_wins.empty else float("nan")
    out["bull_avg_conf_loss"] = float(bull_losses["confidence"].mean()) if not bull_losses.empty else float("nan")
    out["bear_avg_conf_win"] = float(bear_wins["confidence"].mean()) if not bear_wins.empty else float("nan")
    out["bear_avg_conf_loss"] = float(bear_losses["confidence"].mean()) if not bear_losses.empty else float("nan")

    gross_profit_bnb = float(wins["pnl_bnb"].sum()) if not wins.empty else 0.0
    gross_loss_bnb = float(-losses["pnl_bnb"].sum()) if not losses.empty else 0.0
    out["gross_profit_bnb"] = gross_profit_bnb
    out["gross_loss_bnb"] = gross_loss_bnb
    out["profit_factor"] = (gross_profit_bnb / gross_loss_bnb) if gross_loss_bnb > 0 else float("inf")

    out["max_win_streak"] = int(_max_streak((called["pnl_bnb"] > 0).tolist()))
    out["max_loss_streak"] = int(_max_streak((called["pnl_bnb"] < 0).tolist()))

    equity_bnb = called["pnl_bnb"].cumsum()
    running_peak_bnb = equity_bnb.cummax()
    drawdown_bnb = equity_bnb - running_peak_bnb

    out["max_drawdown_bnb"] = float(drawdown_bnb.min()) if not drawdown_bnb.empty else 0.0
    out["max_drawdown_usd"] = out["max_drawdown_bnb"] * bnb_usd_price
    out["max_drawdown_units"] = out["max_drawdown_bnb"] / stake_bnb if stake_bnb > 0 else 0.0

    bucket_stats = _confidence_bucket_stats(called)
    out.update(bucket_stats)

    return out


def _finalize_empty_called_metrics(out: dict) -> dict:
    out["called_accuracy"] = float("nan")
    out["called_bull_accuracy"] = float("nan")
    out["called_bear_accuracy"] = float("nan")

    out["bull_calls"] = 0
    out["bear_calls"] = 0
    out["coverage_bull_pct"] = 0.0
    out["coverage_bear_pct"] = 0.0

    out["bull_net_units"] = 0.0
    out["bear_net_units"] = 0.0
    out["bull_net_bnb"] = 0.0
    out["bear_net_bnb"] = 0.0
    out["bull_net_usd"] = 0.0
    out["bear_net_usd"] = 0.0

    out["gross_wins"] = 0
    out["gross_losses"] = 0

    out["net_units"] = 0.0
    out["net_bnb"] = 0.0
    out["net_usd"] = 0.0

    out["avg_bnb_per_bet"] = float("nan")
    out["avg_usd_per_bet"] = float("nan")

    out["avg_conf_win"] = float("nan")
    out["avg_conf_loss"] = float("nan")
    out["bull_avg_conf"] = float("nan")
    out["bear_avg_conf"] = float("nan")
    out["bull_avg_conf_win"] = float("nan")
    out["bull_avg_conf_loss"] = float("nan")
    out["bear_avg_conf_win"] = float("nan")
    out["bear_avg_conf_loss"] = float("nan")

    out["gross_profit_bnb"] = 0.0
    out["gross_loss_bnb"] = 0.0
    out["profit_factor"] = float("nan")

    out["max_loss_streak"] = 0
    out["max_win_streak"] = 0
    out["max_drawdown_bnb"] = 0.0
    out["max_drawdown_usd"] = 0.0
    out["max_drawdown_units"] = 0.0

    for key in [
        "bucket_50_54_n", "bucket_50_54_acc", "bucket_50_54_net_bnb", "bucket_50_54_net_usd",
        "bucket_54_58_n", "bucket_54_58_acc", "bucket_54_58_net_bnb", "bucket_54_58_net_usd",
        "bucket_58_62_n", "bucket_58_62_acc", "bucket_58_62_net_bnb", "bucket_58_62_net_usd",
        "bucket_62_plus_n", "bucket_62_plus_acc", "bucket_62_plus_net_bnb", "bucket_62_plus_net_usd",
    ]:
        out[key] = float("nan") if key.endswith("_acc") else 0.0 if ("_net_" in key) else 0

    return out


def _confidence_bucket_stats(called: pd.DataFrame) -> dict:
    out: dict = {}

    buckets = [
        ("bucket_50_54", 0.50, 0.54),
        ("bucket_54_58", 0.54, 0.58),
        ("bucket_58_62", 0.58, 0.62),
        ("bucket_62_plus", 0.62, None),
    ]

    for name, lo, hi in buckets:
        if hi is None:
            sub = called[called["confidence"] >= lo]
        else:
            sub = called[(called["confidence"] >= lo) & (called["confidence"] < hi)]

        out[f"{name}_n"] = int(len(sub))
        out[f"{name}_acc"] = float(sub["is_hit"].mean()) if not sub.empty else float("nan")
        out[f"{name}_net_bnb"] = float(sub["pnl_bnb"].sum()) if not sub.empty else 0.0
        out[f"{name}_net_usd"] = float(sub["pnl_usd"].sum()) if not sub.empty else 0.0

    return out


def _max_streak(mask: Iterable[bool]) -> int:
    best = 0
    cur = 0
    for flag in mask:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best