from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import compute_prediction_metrics
from .leaderboard import save_leaderboard
from .feature_blocks import build_feature_matrix
from .rule_debug import save_rule_debug_report
from main.research.epoch_tournament.models.registry import build_model


def run_tournament(
    epoch_df: pd.DataFrame,
    config: dict,
    output_dir: Path,
    return_leaderboard: bool = False,
):
    leaderboard_rows: list[dict] = []
    ran_any_model = False

    for model_cfg in config.get("models", []):
        if not model_cfg.get("enabled", False):
            continue

        ran_any_model = True
        model_id = model_cfg["model_id"]
        print("-" * 100)
        print(f"[model] {model_id} starting")

        pred_df = run_single_model(
            epoch_df=epoch_df,
            config=config,
            model_cfg=model_cfg,
        )

        pred_path = output_dir / f"predictions_{model_id}.csv"
        pred_df.to_csv(pred_path, index=False)

        # Rule-model specific debug report
        if model_cfg.get("classifier") == "rules":
            debug_path = save_rule_debug_report(pred_df, output_dir, model_id)
            print(f"[rule_debug] saved: {debug_path}")

        summary = compute_prediction_metrics(pred_df, config.get("wager", {}))
        summary["model_id"] = model_id
        summary["family"] = model_cfg.get("family", "unknown")
        leaderboard_rows.append(summary)

        summary_df = pd.DataFrame([summary])
        summary_path = output_dir / f"summary_{model_id}.csv"
        summary_df.to_csv(summary_path, index=False)

        print(
            f"[model] {model_id}"
            f" | overall_acc={summary.get('overall_accuracy', float('nan')):.4f}"
            f" | directional_coverage={summary.get('directional_coverage_pct', float('nan')):.2f}%"
            f" | called_acc={summary.get('called_accuracy', float('nan')):.4f}"
            f" | bull_called_acc={summary.get('called_bull_accuracy', float('nan')):.4f}"
            f" | bear_called_acc={summary.get('called_bear_accuracy', float('nan')):.4f}"
            f" | coverage={summary.get('coverage_pct', float('nan')):.2f}%"
            f" | bull_calls={summary.get('bull_calls', 0)}"
            f" | bear_calls={summary.get('bear_calls', 0)}"
            f" | skip_rows={summary.get('skip_rows', 0)}"
            f" | bull_net_usd={summary.get('bull_net_usd', 0.0):.2f}"
            f" | bear_net_usd={summary.get('bear_net_usd', 0.0):.2f}"
            f" | net_bnb={summary.get('net_bnb', 0.0):.5f}"
            f" | net_usd={summary.get('net_usd', 0.0):.2f}"
            f" | profit_factor={summary.get('profit_factor', float('nan')):.3f}"
            f" | max_dd_usd={summary.get('max_drawdown_usd', 0.0):.2f}"
        )

    if not ran_any_model:
        raise ValueError(
            "No enabled models were found in config['models']. "
            "Populate config with at least one enabled model."
        )

    lb_path = save_leaderboard(leaderboard_rows, output_dir)
    print(f"[leaderboard] saved: {lb_path}")

    if return_leaderboard:
        leaderboard_df = pd.DataFrame(leaderboard_rows)
        sort_cols = [c for c in ["net_usd", "called_accuracy", "coverage_pct"] if c in leaderboard_df.columns]
        if sort_cols:
            leaderboard_df = leaderboard_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        return leaderboard_df.reset_index(drop=True)

    return None


def run_single_model(epoch_df: pd.DataFrame, config: dict, model_cfg: dict) -> pd.DataFrame:
    warmup_epochs = int(config["backtest"].get("warmup_epochs", 300))
    min_train_epochs = int(config["backtest"].get("min_train_epochs", 250))
    refit_every_n = int(config["backtest"].get("refit_every_n_epochs", 50))

    model = build_model(model_cfg)
    feature_blocks = model_cfg.get("feature_blocks", [])

    work_df = epoch_df.copy().reset_index(drop=True)

    x_all, feature_cols = build_feature_matrix(work_df, feature_blocks, config)

    # --- DEBUG: confirm segment engine is active ---
    print(f"[debug] model={model_cfg['model_id']} | feature_count={len(feature_cols)}")

    segment_cols = [
        c for c in feature_cols
        if c.startswith(("segment_", "anchor_", "tail_", "seg_"))
           or c in {
               "fan_spread_now",
               "fan_spread_mean",
               "fan_spread_slope",
               "fan_spread_accel",
               "fan_inversion_count",
               "fan_order_violation_now",
               "price_anchor_to_now_delta",
               "tail_price_delta",
           }
    ]

    print(f"[debug] model={model_cfg['model_id']} | feature_count={len(feature_cols)}")
    print(f"[debug] segment_feature_count={len(segment_cols)}")
    print(f"[debug] segment_cols_sample={segment_cols[:20]}")
    anchor_cols = [c for c in feature_cols if c.startswith(("anchor_", "tail_", "seg_"))]
    print(f"[debug] anchor_tail_seg_feature_count={len(anchor_cols)}")
    print(f"[debug] anchor_tail_seg_cols_sample={anchor_cols[:20]}")
    # ---------------------------------------------

    work_df = pd.concat([work_df, x_all], axis=1)

    predictions: list[dict] = []
    last_fit_idx = None

    for i in range(warmup_epochs, len(work_df) - 1):
        train_df = work_df.iloc[:i].copy()
        test_row = work_df.iloc[[i]].copy()

        if len(train_df) < min_train_epochs:
            continue

        need_refit = (last_fit_idx is None) or ((i - last_fit_idx) >= refit_every_n)
        if need_refit:
            x_train = train_df[feature_cols].fillna(0.0)
            y_train = train_df["target_label"].astype(int)
            model.fit(x_train, y_train)
            last_fit_idx = i

        x_test = test_row[feature_cols].fillna(0.0)
        pred = model.predict_one(x_test)

        if pred.trend == "Bull":
            pred_label = 1
        elif pred.trend == "Bear":
            pred_label = 0
        else:
            pred_label = -1

        predictions.append(
            {
                "epoch": int(test_row.iloc[0]["epoch"]),
                "scheduled_bet_time": test_row.iloc[0]["scheduled_bet_time"],
                "next_epoch_time": test_row.iloc[0]["next_epoch_time"],
                "pred_trend": pred.trend,
                "pred_label": pred_label,
                "actual_trend": test_row.iloc[0]["target_trend"],
                "actual_label": int(test_row.iloc[0]["target_label"]),
                "prob_bull": pred.prob_bull,
                "prob_bear": pred.prob_bear,
                "confidence": pred.confidence,
                "wager": bool(pred.wager),
                "reason": pred.reason,
            }
        )

    return pd.DataFrame(predictions)


def train_final_model_from_best_row(epoch_df: pd.DataFrame, config: dict, best_model_id: str):
    model_cfg = next(m for m in config.get("models", []) if m["model_id"] == best_model_id)
    model = build_model(model_cfg)

    work_df = epoch_df.copy().reset_index(drop=True)
    x_all, feature_cols = build_feature_matrix(
        work_df,
        model_cfg.get("feature_blocks", []),
        config,
    )
    work_df = pd.concat([work_df, x_all], axis=1)

    x_train = work_df[feature_cols].fillna(0.0)
    y_train = work_df["target_label"].astype(int)
    model.fit(x_train, y_train)

    return model, feature_cols, model_cfg