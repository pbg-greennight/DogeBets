from __future__ import annotations

import pandas as pd

from .common_v21 import clip01


def add_method_v21_composites(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()

    df["v21_bull_continuation_score"] = (
        0.32 * df["v21_msbc_bull_alignment"]
        + 0.22 * (df["v21_hyst_continuation_quality"] * (df["v21_hyst_primary_sign_num"] > 0).astype(float))
        + 0.18 * df["v21_gcs_bull_alignment"]
        + 0.18 * df["v21_motion_trend_quality_bull"]
        + 0.10 * (1.0 - df["v21_msbc_fragility"])
    ).map(clip01)

    df["v21_bear_continuation_score"] = (
        0.32 * df["v21_msbc_bear_alignment"]
        + 0.22 * (df["v21_hyst_continuation_quality"] * (df["v21_hyst_primary_sign_num"] < 0).astype(float))
        + 0.18 * df["v21_gcs_bear_alignment"]
        + 0.18 * df["v21_motion_trend_quality_bear"]
        + 0.10 * (1.0 - df["v21_msbc_fragility"])
    ).map(clip01)

    df["v21_bull_exhaustion_score"] = (
        0.28 * (df["v21_hyst_frozen_exhaustion"] * (df["v21_hyst_primary_sign_num"] > 0).astype(float))
        + 0.20 * (df["v21_msbc_fragility"] * (df["v21_msbc_transfer_dir_num"] > 0).astype(float))
        + 0.16 * (df["v21_gbc_fast_hook_instability"] * (df["v21_msbc_transfer_dir_num"] > 0).astype(float))
        + 0.16 * (df["v21_gbc_slow_flattening"] * (df["v21_msbc_transfer_dir_num"] > 0).astype(float))
        + 0.10 * df["v21_channel_fast_oscillation"]
        + 0.10 * (df["v21_channel_reexpansion_failure"] * (df["v21_msbc_transfer_dir_num"] > 0).astype(float))
    ).map(clip01)

    df["v21_bear_exhaustion_score"] = (
        0.28 * (df["v21_hyst_frozen_exhaustion"] * (df["v21_hyst_primary_sign_num"] < 0).astype(float))
        + 0.20 * (df["v21_msbc_fragility"] * (df["v21_msbc_transfer_dir_num"] < 0).astype(float))
        + 0.16 * (df["v21_gbc_fast_hook_instability"] * (df["v21_msbc_transfer_dir_num"] < 0).astype(float))
        + 0.16 * (df["v21_gbc_slow_flattening"] * (df["v21_msbc_transfer_dir_num"] < 0).astype(float))
        + 0.10 * df["v21_channel_fast_oscillation"]
        + 0.10 * (df["v21_channel_reexpansion_failure"] * (df["v21_msbc_transfer_dir_num"] < 0).astype(float))
    ).map(clip01)

    mixed_margin = config["v21"]["mixed_dir_margin"]
    df["v21_conflict_score"] = (
        0.30 * df["v21_msbc_fast_slow_conflict"]
        + 0.25 * df["v21_gcs_positional_conflict"]
        + 0.20 * df["v21_hyst_instability_risk"]
        + 0.15 * df["v21_channel_fast_oscillation"]
        + 0.10 * (abs(df["v21_msbc_bull_alignment"] - df["v21_msbc_bear_alignment"]) < mixed_margin).astype(float)
    ).map(clip01)

    df["v21_compression_trap_score"] = (
        0.35 * df["v21_gcs_contraction_trap"]
        + 0.25 * df["v21_channel_compression_persistence"]
        + 0.20 * df["v21_channel_reexpansion_failure"]
        + 0.20 * (0.50 * df["v21_channel_fast_oscillation"] + 0.50 * (df["v21_msbc_fast_slow_conflict"] > config["v21"]["fast_slow_warn"]).astype(float))
    ).map(clip01)

    df["v21_bull_invalidators"] = (
        0.45 * df["v21_bear_continuation_score"]
        + 0.25 * (df["v21_msbc_slow_dir_mean"] < -0.5).astype(float)
        + 0.20 * (df["v21_gcs_slow_pos_mean"] < -config["gcs"]["slow_confirm_pos_thresh"]).astype(float)
        + 0.10 * df["v21_compression_trap_score"]
    ).map(clip01)

    df["v21_bear_invalidators"] = (
        0.45 * df["v21_bull_continuation_score"]
        + 0.25 * (df["v21_msbc_slow_dir_mean"] > 0.5).astype(float)
        + 0.20 * (df["v21_gcs_slow_pos_mean"] > config["gcs"]["slow_confirm_pos_thresh"]).astype(float)
        + 0.10 * df["v21_compression_trap_score"]
    ).map(clip01)

    df["v21_reversal_to_bull_bonus"] = (
        0.40 * df["v21_bear_exhaustion_score"]
        + 0.25 * ((df["v21_msbc_fast_dir_mean"] > 0) & (df["v21_msbc_slow_dir_mean"] <= 0)).astype(float)
        + 0.20 * (df["v21_gcs_fast_slow_pos_gap"] > 0).astype(float)
        + 0.15 * (1.0 - df["v21_compression_trap_score"])
    ).map(clip01)

    df["v21_reversal_to_bear_bonus"] = (
        0.40 * df["v21_bull_exhaustion_score"]
        + 0.25 * ((df["v21_msbc_fast_dir_mean"] < 0) & (df["v21_msbc_slow_dir_mean"] >= 0)).astype(float)
        + 0.20 * (df["v21_gcs_fast_slow_pos_gap"] < 0).astype(float)
        + 0.15 * (1.0 - df["v21_compression_trap_score"])
    ).map(clip01)

    return df
