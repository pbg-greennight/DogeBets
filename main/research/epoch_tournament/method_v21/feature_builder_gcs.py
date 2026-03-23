from __future__ import annotations

import pandas as pd

from .common_v21 import GCS_REGIME_MAP, SIGMAS_ALL, clip01, clip11, mean_safe, safe_div, TRANSFER_DIR_MAP, TRANSFER_STATE_MAP


def _ensure(df: pd.DataFrame, col: str, default=float("nan")) -> None:
    if col not in df.columns:
        df[col] = default


def add_gcs_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    df["v21_src_missing_gcs"] = 0.0

    for sigma in SIGMAS_ALL:
        for c in [f"src_gcs_regime_s{sigma}", f"src_gcs_pos_s{sigma}", f"src_gcs_width_s{sigma}", f"src_gcs_mid_slope_s{sigma}"]:
            _ensure(df, c)
        df[f"v21_gcs_regime_num_s{sigma}"] = df[f"src_gcs_regime_s{sigma}"].map(lambda x: GCS_REGIME_MAP.get(str(x).lower(), 0.0))
        df[f"v21_gcs_pos_s{sigma}"] = df[f"src_gcs_pos_s{sigma}"]
        df[f"v21_gcs_width_s{sigma}"] = df[f"src_gcs_width_s{sigma}"]
        df[f"v21_gcs_mid_slope_s{sigma}"] = df[f"src_gcs_mid_slope_s{sigma}"]

    needed = [
        "src_gcs_regime_contract_count", "src_gcs_regime_expand_count", "src_gcs_regime_flat_count", "src_gcs_regime_contract_all",
        "src_gcs_spacing_state", "src_gcs_fan_state", "src_gcs_transfer_dir", "src_gcs_transfer_depth", "src_gcs_transfer_state",
    ]
    for c in needed:
        _ensure(df, c)

    df["v21_gcs_regime_contract_count_norm"] = df["src_gcs_regime_contract_count"].map(lambda x: clip01(safe_div(x, 6.0)))
    df["v21_gcs_regime_expand_count_norm"] = df["src_gcs_regime_expand_count"].map(lambda x: clip01(safe_div(x, 6.0)))
    df["v21_gcs_regime_flat_count_norm"] = df["src_gcs_regime_flat_count"].map(lambda x: clip01(safe_div(x, 6.0)))
    df["v21_gcs_regime_contract_all"] = df["src_gcs_regime_contract_all"].fillna(0.0).astype(float)
    df["v21_gcs_transfer_dir_num"] = df["src_gcs_transfer_dir"].map(lambda x: TRANSFER_DIR_MAP.get(str(x).lower(), 0.0))
    df["v21_gcs_transfer_depth_norm"] = df["src_gcs_transfer_depth"].map(lambda x: clip01(safe_div(x, 5.0)))
    df["v21_gcs_transfer_state_norm"] = df["src_gcs_transfer_state"].map(lambda x: TRANSFER_STATE_MAP.get(str(x).lower(), 0.0))
    df["v21_gcs_fast_pos_mean"] = df.apply(lambda r: mean_safe([r["v21_gcs_pos_s8"], r["v21_gcs_pos_s23"]]), axis=1)
    df["v21_gcs_slow_pos_mean"] = df.apply(lambda r: mean_safe([r["v21_gcs_pos_s68"], r["v21_gcs_pos_s83"]]), axis=1)
    df["v21_gcs_fast_slow_pos_gap"] = (df["v21_gcs_fast_pos_mean"] - df["v21_gcs_slow_pos_mean"]).map(clip11)

    def _pos_conflict(r: pd.Series) -> float:
        fast_slow_gap_abs = abs(r["v21_gcs_fast_slow_pos_gap"])
        mixed_spacing = float(str(r["src_gcs_spacing_state"]).lower() in ["mixed", "split", "uneven"])
        split_transfer = float(str(r["src_gcs_transfer_state"]).lower() in ["none", "split", "shallow"])
        return clip01(0.40 * fast_slow_gap_abs + 0.25 * r["v21_gcs_regime_contract_count_norm"] + 0.20 * mixed_spacing + 0.15 * split_transfer)

    def _trap(r: pd.Series) -> float:
        fast_only_move = float(abs(r["v21_gcs_fast_pos_mean"]) > config["gcs"]["fast_only_pos_thresh"] and abs(r["v21_gcs_slow_pos_mean"]) < config["gcs"]["slow_confirm_pos_thresh"])
        slow_nonconfirm = float(abs(r["v21_gcs_slow_pos_mean"]) < config["gcs"]["slow_confirm_pos_thresh"])
        mixed_fan = float(str(r["src_gcs_fan_state"]).lower() in ["mixed", "uneven", "split"])
        shallow_or_none = float(str(r["src_gcs_transfer_state"]).lower() in ["none", "shallow", "split"])
        return clip01(0.35 * r["v21_gcs_regime_contract_all"] + 0.20 * shallow_or_none + 0.20 * fast_only_move + 0.15 * slow_nonconfirm + 0.10 * mixed_fan)

    def _bull_align(r: pd.Series) -> float:
        above_mid_agree = mean_safe([float(r[f"v21_gcs_pos_s{s}"] > 0) for s in [23, 38, 53, 68, 83]])
        transfer_up = float(r["v21_gcs_transfer_dir_num"] > 0)
        non_trap = 1.0 - r["v21_gcs_contraction_trap"]
        fan_ok = float(str(r["src_gcs_fan_state"]).lower() in ["up", "fanning_up", "ordered_up"])
        return clip01(0.30 * transfer_up + 0.20 * (transfer_up * r["v21_gcs_transfer_depth_norm"]) + 0.25 * above_mid_agree + 0.15 * fan_ok + 0.10 * non_trap)

    def _bear_align(r: pd.Series) -> float:
        below_mid_agree = mean_safe([float(r[f"v21_gcs_pos_s{s}"] < 0) for s in [23, 38, 53, 68, 83]])
        transfer_down = float(r["v21_gcs_transfer_dir_num"] < 0)
        non_trap = 1.0 - r["v21_gcs_contraction_trap"]
        fan_ok = float(str(r["src_gcs_fan_state"]).lower() in ["down", "fanning_down", "ordered_down"])
        return clip01(0.30 * transfer_down + 0.20 * (transfer_down * r["v21_gcs_transfer_depth_norm"]) + 0.25 * below_mid_agree + 0.15 * fan_ok + 0.10 * non_trap)

    df["v21_gcs_positional_conflict"] = df.apply(_pos_conflict, axis=1)
    df["v21_gcs_contraction_trap"] = df.apply(_trap, axis=1)
    df["v21_gcs_bull_alignment"] = df.apply(_bull_align, axis=1)
    df["v21_gcs_bear_alignment"] = df.apply(_bear_align, axis=1)
    return df
