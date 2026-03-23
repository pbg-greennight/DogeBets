from __future__ import annotations

import json
from pathlib import Path

from backtest.parquet_loader import load_parquet_data
from backtest.epoch_builder import prepare_epoch_dataframe
from backtest.feature_blocks import build_feature_matrix
from backtest.artifact_io import load_model_artifact
from backtest.replay_runner import run_locked_replay


def main() -> int:
    root = Path(__file__).resolve().parent
    cfg_path = root / "config" / "replay_config.json"
    output_dir = root / "outputs"
    artifact_path = root / "artifacts" / "best_model_v1.pkl"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_payload = load_model_artifact(artifact_path)
    raw_df = load_parquet_data(cfg["data"]["parquet_path"], cfg["data"])
    epoch_df = prepare_epoch_dataframe(raw_df, cfg)

    x_all, _ = build_feature_matrix(epoch_df.copy(), artifact_payload["model_cfg"].get("feature_blocks", []), cfg)
    epoch_df = epoch_df.copy()
    epoch_df = epoch_df.join(x_all)

    print("=" * 100)
    print("Replay Evaluator Starting")
    print(f"[mode] {cfg.get('mode')}")
    print(f"[offset] {cfg['backtest']['decision_offset_sec']}")
    print(f"[artifact] {artifact_path}")
    print("=" * 100)

    _, summary = run_locked_replay(epoch_df, artifact_payload, cfg, output_dir)

    print(
        f"[replay] model={summary.get('model_id')}"
        f" | called_acc={summary.get('called_accuracy', float('nan')):.4f}"
        f" | coverage={summary.get('coverage_pct', float('nan')):.2f}%"
        f" | net_usd={summary.get('net_usd', 0.0):.2f}"
        f" | max_dd_usd={summary.get('max_drawdown_usd', 0.0):.2f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())