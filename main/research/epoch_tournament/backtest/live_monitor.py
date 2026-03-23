from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Any

import pandas as pd

from .feature_blocks import build_feature_matrix
from .metrics import compute_prediction_metrics


def run_live_monitor(
    raw_df_loader: Callable[[], pd.DataFrame],
    prepare_epoch_dataframe: Callable[[pd.DataFrame, dict], pd.DataFrame],
    artifact_payload: dict,
    config: dict,
    output_dir: Path,
) -> None:
    model = artifact_payload["model"]
    feature_cols = artifact_payload["feature_cols"]
    model_cfg = artifact_payload["model_cfg"]

    poll_seconds = int(config.get("live", {}).get("poll_seconds", 30))

    live_log_path = output_dir / "live_monitor_log.csv"
    summary_path = output_dir / "live_monitor_summary.csv"

    seen_epochs: set[int] = set()
    all_rows: list[dict] = []

    print("[live] live monitor started")

    while True:
        try:
            raw_df = raw_df_loader()
            epoch_df = prepare_epoch_dataframe(raw_df, config)
            work_df = epoch_df.copy().reset_index(drop=True)

            x_all, _built_feature_cols = build_feature_matrix(
                work_df,
                model_cfg.get("feature_blocks", []),
                config,
            )
            work_df = pd.concat([work_df, x_all], axis=1)

            for i in range(len(work_df) - 1):
                epoch_id = int(work_df.iloc[i]["epoch"])
                if epoch_id in seen_epochs:
                    continue

                x_test = work_df.iloc[[i]][feature_cols].fillna(0.0)
                pred = model.predict_one(x_test)

                row = {
                    "epoch": epoch_id,
                    "scheduled_bet_time": work_df.iloc[i]["scheduled_bet_time"],
                    "next_epoch_time": work_df.iloc[i]["next_epoch_time"],
                    "pred_trend": pred.trend,
                    "pred_label": 1 if pred.trend == "Bull" else 0,
                    "actual_trend": work_df.iloc[i]["target_trend"],
                    "actual_label": int(work_df.iloc[i]["target_label"]),
                    "prob_bull": pred.prob_bull,
                    "prob_bear": pred.prob_bear,
                    "confidence": pred.confidence,
                    "wager": pred.wager,
                    "reason": pred.reason,
                }

                all_rows.append(row)
                seen_epochs.add(epoch_id)

            if all_rows:
                log_df = pd.DataFrame(all_rows)
                log_df.to_csv(live_log_path, index=False)

                summary = compute_prediction_metrics(log_df, config.get("wager", {}))
                pd.DataFrame([summary]).to_csv(summary_path, index=False)

                print(
                    f"[live] epochs_logged={len(log_df)}"
                    f" | called_acc={summary.get('called_accuracy', float('nan')):.4f}"
                    f" | coverage={summary.get('coverage_pct', float('nan')):.2f}%"
                    f" | net_usd={summary.get('net_usd', 0.0):.2f}"
                    f" | max_dd_usd={summary.get('max_drawdown_usd', 0.0):.2f}"
                )

        except Exception as e:
            print(f"[live] error: {e}")

        time.sleep(poll_seconds)