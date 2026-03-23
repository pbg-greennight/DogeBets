from __future__ import annotations

import json
import os
from pathlib import Path

from backtest.parquet_loader import load_parquet_data
from backtest.epoch_builder import prepare_epoch_dataframe
from backtest.walkforward_runner import run_tournament


def main() -> int:
    root = Path(__file__).resolve().parent
    cfg_path = root / "config" / "tournament_config.json"

    if not cfg_path.exists():
        print(f"[error] Missing config: {cfg_path}")
        return 1

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    parquet_path = cfg["data"]["parquet_path"]
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Offline Epoch Tournament Starting")
    print(f"[config] parquet_path = {parquet_path}")
    print(f"[config] output_dir   = {output_dir}")
    print("=" * 100)

    raw_df = load_parquet_data(parquet_path, cfg["data"])
    epoch_df = prepare_epoch_dataframe(raw_df, cfg)

    print(f"[data] rows={len(raw_df):,} | epochs={len(epoch_df):,}")

    run_tournament(
        epoch_df=epoch_df,
        config=cfg,
        output_dir=output_dir,
    )

    print("[done] Tournament complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())