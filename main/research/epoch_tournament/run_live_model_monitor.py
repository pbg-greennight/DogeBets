from __future__ import annotations

import json
from pathlib import Path

from backtest.parquet_loader import load_parquet_data
from backtest.epoch_builder import prepare_epoch_dataframe
from backtest.artifact_io import load_model_artifact
from backtest.live_monitor import run_live_monitor


def main() -> int:
    root = Path(__file__).resolve().parent
    cfg_path = root / "config" / "live_monitor_config.json"
    output_dir = root / "outputs"
    artifact_path = root / "artifacts" / "best_model_v1.pkl"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_payload = load_model_artifact(artifact_path)

    def raw_df_loader():
        return load_parquet_data(cfg["data"]["parquet_path"], cfg["data"])

    print("=" * 100)
    print("Live Model Monitor Starting")
    print(f"[mode] {cfg.get('mode')}")
    print(f"[offset] {cfg['backtest']['decision_offset_sec']}")
    print(f"[artifact] {artifact_path}")
    print("=" * 100)

    run_live_monitor(raw_df_loader, prepare_epoch_dataframe, artifact_payload, cfg, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())