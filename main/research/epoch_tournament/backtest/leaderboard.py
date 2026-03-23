from __future__ import annotations

from pathlib import Path

import pandas as pd


LEADERBOARD_ORDER = [
    "model_id",
    "family",

    "epochs_seen",
    "calls_made",
    "coverage_pct",
    "coverage_bull_pct",
    "coverage_bear_pct",

    "overall_accuracy",
    "called_accuracy",
    "called_bull_accuracy",
    "called_bear_accuracy",

    "bull_precision",
    "bull_recall",
    "bear_precision",
    "bear_recall",

    "avg_confidence",
    "avg_conf_win",
    "avg_conf_loss",
    "bull_avg_conf",
    "bear_avg_conf",
    "bull_avg_conf_win",
    "bull_avg_conf_loss",
    "bear_avg_conf_win",
    "bear_avg_conf_loss",

    "brier_score",
    "log_loss",

    "bull_calls",
    "bear_calls",

    "stake_bnb",
    "bnb_usd_price",
    "payout_win_multiple",
    "site_fee_on_win_rate",
    "gas_bet_usd",
    "gas_claim_usd",

    "bull_net_units",
    "bear_net_units",
    "net_units",

    "bull_net_bnb",
    "bear_net_bnb",
    "net_bnb",

    "bull_net_usd",
    "bear_net_usd",
    "net_usd",

    "avg_bnb_per_bet",
    "avg_usd_per_bet",

    "gross_wins",
    "gross_losses",
    "gross_profit_bnb",
    "gross_loss_bnb",
    "profit_factor",

    "max_loss_streak",
    "max_win_streak",
    "max_drawdown_units",
    "max_drawdown_bnb",
    "max_drawdown_usd",

    "tn_bear_ok",
    "fp_false_bull",
    "fn_false_bear",
    "tp_bull_ok",

    "bucket_50_54_n",
    "bucket_50_54_acc",
    "bucket_50_54_net_bnb",
    "bucket_50_54_net_usd",

    "bucket_54_58_n",
    "bucket_54_58_acc",
    "bucket_54_58_net_bnb",
    "bucket_54_58_net_usd",

    "bucket_58_62_n",
    "bucket_58_62_acc",
    "bucket_58_62_net_bnb",
    "bucket_58_62_net_usd",

    "bucket_62_plus_n",
    "bucket_62_plus_acc",
    "bucket_62_plus_net_bnb",
    "bucket_62_plus_net_usd",
]


def save_leaderboard(rows: list[dict], output_dir: Path) -> Path:
    df = pd.DataFrame(rows)

    out_path = output_dir / "leaderboard.csv"

    if df.empty:
        df.to_csv(out_path, index=False)
        return out_path

    cols = [c for c in LEADERBOARD_ORDER if c in df.columns] + [c for c in df.columns if c not in LEADERBOARD_ORDER]
    df = df[cols]

    sort_cols = [c for c in ["net_usd", "called_accuracy", "coverage_pct"] if c in df.columns]
    if sort_cols:
        ascending = [False] * len(sort_cols)
        df = df.sort_values(sort_cols, ascending=ascending)

    df.to_csv(out_path, index=False)
    return out_path