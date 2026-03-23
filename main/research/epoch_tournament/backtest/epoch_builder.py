from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def prepare_epoch_dataframe(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Build one row per epoch using ONLY bars available up to:

        cutoff_time = next_epoch_time - decision_offset_sec

    This makes the dataset offset-aware, so:
    - decision_offset_sec = 0 uses bars right up to the epoch boundary
    - decision_offset_sec = 12 stops 12 seconds early

    Expected canonical input columns after parquet_loader rename:
    - timestamp
    - epoch
    - next_epoch_time
    - open
    - high
    - low
    - close
    - volume
    - vwap

    Output includes:
    - partial-epoch OHLCV snapshot up to cutoff
    - bar-count diagnostics
    - coverage diagnostics
    - target label/trend for next epoch
    """
    if raw_df.empty:
        return pd.DataFrame()

    decision_offset_sec = int(config.get("backtest", {}).get("decision_offset_sec", 12))

    df = raw_df.copy()

    # Defensive typing
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["next_epoch_time"] = pd.to_datetime(df["next_epoch_time"], errors="coerce", utc=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    if "vwap" in df.columns:
        numeric_cols.append("vwap")

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["epoch", "timestamp"]).reset_index(drop=True)

    rows: list[dict] = []

    for epoch_id, g in df.groupby("epoch", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)

        # Need a usable next_epoch_time
        next_epoch_candidates = g["next_epoch_time"].dropna()
        if next_epoch_candidates.empty:
            continue

        next_epoch_time = next_epoch_candidates.iloc[-1]
        cutoff_time = next_epoch_time - pd.Timedelta(seconds=decision_offset_sec)

        usable = g[g["timestamp"] <= cutoff_time].copy()
        bars_total = int(len(g))
        bars_used = int(len(usable))

        # No information available before cutoff -> skip epoch
        if usable.empty:
            continue

        # Epoch start timestamp from first usable bar
        ts_start = usable["timestamp"].iloc[0]
        ts_end_used = usable["timestamp"].iloc[-1]

        # Full epoch start/end diagnostics from all bars
        full_ts_start = g["timestamp"].iloc[0]
        full_ts_end = g["timestamp"].iloc[-1]

        # Partial snapshot up to cutoff
        open_used = float(usable["open"].iloc[0])
        high_used = float(usable["high"].max())
        low_used = float(usable["low"].min())
        close_used = float(usable["close"].iloc[-1])
        volume_used = float(usable["volume"].sum())

        # VWAP fallback from usable bars if not already meaningful
        if "vwap" in usable.columns and usable["vwap"].notna().any():
            vwap_used = float(usable["vwap"].iloc[-1])
        else:
            vol_sum = float(usable["volume"].sum())
            if vol_sum > EPS:
                vwap_used = float((usable["close"] * usable["volume"]).sum() / vol_sum)
            else:
                vwap_used = float(close_used)

        # Timing diagnostics
        seconds_to_boundary_from_last_bar = float((next_epoch_time - ts_end_used).total_seconds())
        seconds_of_visible_epoch = float((ts_end_used - ts_start).total_seconds()) if bars_used >= 2 else 0.0
        full_epoch_visible_seconds = float((full_ts_end - full_ts_start).total_seconds()) if bars_total >= 2 else 0.0

        # Coverage diagnostics
        coverage_ratio = float(bars_used / max(bars_total, 1))
        bars_missing = int(max(0, bars_total - bars_used))

        # Tail / pre-boundary quick diagnostics
        last_3 = usable.tail(min(3, bars_used))
        last_5 = usable.tail(min(5, bars_used))
        last_10 = usable.tail(min(10, bars_used))

        tail_ret_3 = float((last_3["close"].iloc[-1] / last_3["close"].iloc[0] - 1.0)) if len(last_3) >= 2 else 0.0
        tail_ret_5 = float((last_5["close"].iloc[-1] / last_5["close"].iloc[0] - 1.0)) if len(last_5) >= 2 else 0.0
        tail_ret_10 = float((last_10["close"].iloc[-1] / last_10["close"].iloc[0] - 1.0)) if len(last_10) >= 2 else 0.0

        row = {
            "epoch": int(epoch_id),
            "timestamp": ts_end_used,
            "ts_start_used": ts_start,
            "ts_end_used": ts_end_used,
            "full_ts_start": full_ts_start,
            "full_ts_end": full_ts_end,
            "next_epoch_time": next_epoch_time,
            "scheduled_bet_time": cutoff_time,
            "decision_offset_sec": decision_offset_sec,

            "open": open_used,
            "high": high_used,
            "low": low_used,
            "close": close_used,
            "volume": volume_used,
            "vwap": vwap_used,

            "bars_used": bars_used,
            "bars_total": bars_total,
            "bars_missing": bars_missing,
            "coverage_ratio": coverage_ratio,

            "seconds_to_boundary_from_last_bar": seconds_to_boundary_from_last_bar,
            "seconds_of_visible_epoch": seconds_of_visible_epoch,
            "full_epoch_visible_seconds": full_epoch_visible_seconds,

            "tail_ret_3": tail_ret_3,
            "tail_ret_5": tail_ret_5,
            "tail_ret_10": tail_ret_10,
        }

        rows.append(row)

    epoch_df = pd.DataFrame(rows)

    if epoch_df.empty:
        return epoch_df

    epoch_df = epoch_df.sort_values("epoch").reset_index(drop=True)

    # Build NEXT-EPOCH target from the next row's close
    epoch_df["next_close"] = epoch_df["close"].shift(-1)
    epoch_df["target_return"] = (epoch_df["next_close"] / epoch_df["close"]) - 1.0
    epoch_df["target_label"] = np.where(epoch_df["target_return"] > 0, 1, 0)
    epoch_df["target_trend"] = np.where(epoch_df["target_label"] == 1, "Bull", "Bear")

    # Optional useful diagnostics
    epoch_df["close_to_vwap"] = (epoch_df["close"] - epoch_df["vwap"]) / (epoch_df["close"].abs() + EPS)
    epoch_df["epoch_range"] = epoch_df["high"] - epoch_df["low"]
    epoch_df["body"] = epoch_df["close"] - epoch_df["open"]
    epoch_df["body_pct"] = epoch_df["body"] / (epoch_df["open"].abs() + EPS)

    # Last row has no next target
    epoch_df = epoch_df.dropna(subset=["next_close", "target_return"]).reset_index(drop=True)

    # Console diagnostics so you can verify offset effects quickly
    print(
        "[epoch_builder]"
        f" offset={decision_offset_sec}s"
        f" | rows={len(epoch_df):,}"
        f" | avg_bars_used={epoch_df['bars_used'].mean():.2f}"
        f" | avg_bars_total={epoch_df['bars_total'].mean():.2f}"
        f" | avg_coverage={epoch_df['coverage_ratio'].mean():.4f}"
        f" | avg_sec_to_boundary={epoch_df['seconds_to_boundary_from_last_bar'].mean():.2f}"
    )

    return epoch_df