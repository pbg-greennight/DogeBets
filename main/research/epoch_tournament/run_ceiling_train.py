from __future__ import annotations

import json
from pathlib import Path

from backtest.parquet_loader import load_parquet_data
from backtest.epoch_builder import prepare_epoch_dataframe
from backtest.walkforward_runner import run_tournament, train_final_model_from_best_row
from backtest.artifact_io import save_model_artifact


def main() -> int:
    root = Path(__file__).resolve().parent
    cfg_path = root / "config" / "ceiling_train_config.json"
    output_dir = root / "outputs"
    artifact_dir = root / "artifacts"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_parquet_data(cfg["data"]["parquet_path"], cfg["data"])
    epoch_df = prepare_epoch_dataframe(raw_df, cfg)

    print("=" * 100)
    print("Ceiling Trainer Starting")
    print(f"[mode] {cfg.get('mode')}")
    print(f"[offset] {cfg['backtest']['decision_offset_sec']}")
    print(f"[data] rows={len(raw_df):,} | epochs={len(epoch_df):,}")
    print("=" * 100)

    leaderboard_df = run_tournament(epoch_df=epoch_df, config=cfg, output_dir=output_dir, return_leaderboard=True)
    best_row = leaderboard_df.iloc[0].to_dict()

    final_model, feature_cols, best_model_cfg = train_final_model_from_best_row(epoch_df, cfg, best_row["model_id"])

    metadata = save_model_artifact(
        model=final_model,
        feature_cols=feature_cols,
        model_cfg=best_model_cfg,
        config=cfg,
        artifact_dir=artifact_dir,
        artifact_name="best_model_v1",
    )

    print(f"[artifact] saved: {metadata['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())