from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import compute_prediction_metrics


def run_locked_replay(epoch_df: pd.DataFrame, artifact_payload: dict, config: dict, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    model = artifact_payload["model"]
    feature_cols = artifact_payload["feature_cols"]
    model_cfg = artifact_payload["model_cfg"]

    work_df = epoch_df.copy().reset_index(drop=True)
    predictions: list[dict] = []

    for i in range(len(work_df) - 1):
        test_row = work_df.iloc[[i]].copy()
        x_test = test_row[feature_cols].fillna(0.0)
        pred = model.predict_one(x_test)

        predictions.append(
            {
                "epoch": int(test_row.iloc[0]["epoch"]),
                "scheduled_bet_time": test_row.iloc[0]["scheduled_bet_time"],
                "next_epoch_time": test_row.iloc[0]["next_epoch_time"],
                "pred_trend": pred.trend,
                "pred_label": 1 if pred.trend == "Bull" else 0,
                "actual_trend": test_row.iloc[0]["target_trend"],
                "actual_label": int(test_row.iloc[0]["target_label"]),
                "prob_bull": pred.prob_bull,
                "prob_bear": pred.prob_bear,
                "confidence": pred.confidence,
                "wager": pred.wager,
                "reason": pred.reason,
                "artifact_model_id": model_cfg.get("model_id", "unknown"),
            }
        )

    pred_df = pd.DataFrame(predictions)
    summary = compute_prediction_metrics(pred_df, config.get("wager", {}))
    summary["model_id"] = model_cfg.get("model_id", "unknown")
    summary["family"] = model_cfg.get("family", "unknown")

    pred_path = output_dir / f"replay_predictions_{summary['model_id']}.csv"
    pred_df.to_csv(pred_path, index=False)

    summary_path = output_dir / f"replay_summary_{summary['model_id']}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    return pred_df, summary