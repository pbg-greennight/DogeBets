from __future__ import annotations

import pandas as pd


REQUIRED_KEYS = [
    "timestamp_col",
    "epoch_col",
    "next_epoch_time_col",
    "open_col",
    "high_col",
    "low_col",
    "close_col",
    "volume_col",
]


def load_parquet_data(parquet_path: str, data_cfg: dict) -> pd.DataFrame:
    for key in REQUIRED_KEYS:
        if key not in data_cfg:
            raise KeyError(f"Missing required config key: {key}")

    df = pd.read_parquet(parquet_path).copy()

    rename_map = {
        data_cfg["timestamp_col"]: "timestamp",
        data_cfg["epoch_col"]: "epoch",
        data_cfg["next_epoch_time_col"]: "next_epoch_time",
        data_cfg["open_col"]: "open",
        data_cfg["high_col"]: "high",
        data_cfg["low_col"]: "low",
        data_cfg["close_col"]: "close",
        data_cfg["volume_col"]: "volume",
    }

    if "vwap_col" in data_cfg and data_cfg["vwap_col"] in df.columns:
        rename_map[data_cfg["vwap_col"]] = "vwap"

    df = df.rename(columns=rename_map)

    required_cols = ["timestamp", "epoch", "next_epoch_time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after rename: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["next_epoch_time"] = pd.to_datetime(df["next_epoch_time"])

    numeric_cols = ["open", "high", "low", "close", "volume"]
    if "vwap" in df.columns:
        numeric_cols.append("vwap")

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["timestamp", "epoch"]).reset_index(drop=True)
    return df