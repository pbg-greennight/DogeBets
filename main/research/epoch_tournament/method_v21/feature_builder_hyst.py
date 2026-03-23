from __future__ import annotations

import pandas as pd

from .common_v21 import (
    PRESSURE_MAP,
    SPREAD_STATE_MAP,
    STATE_RISK_MAP,
    bool01,
    clip01,
    clip11,
    mean_safe,
    safe_div,
    sign3,
)


def _ensure(df: pd.DataFrame, col: str, default=float("nan")) -> None:
    if col not in df.columns:
        df[col] = default


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def add_hyst_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    df["v21_src_missing_hyst"] = 0.0

    needed = [
        "src_hyst_primary_sign",
        "src_hyst_probe_sign",
        "src_hyst_episode_age_sec",
        "src_hyst_last_cross_age_sec",
        "src_hyst_probe_flip_watch",
        "src_hyst_fast_collapse",
        "src_hyst_s0_state",
        "src_hyst_s0_pressure",
        "src_hyst_s0_risk",
        "src_hyst_s0_stability",
        "src_hyst_s0_near_cross",
        "src_hyst_s0_spread_slope",
        "src_hyst_s0_spread_accel",
        "src_hyst_s1_state",
        "src_hyst_s1_pressure",
        "src_hyst_s1_risk",
        "src_hyst_s1_stability",
        "src_hyst_s1_near_cross",
        "src_hyst_s1_spread_slope",
        "src_hyst_s1_spread_accel",
        "src_hyst_leader_max_sigma",
        "src_hyst_leader_min_sigma",
        "src_hyst_ladder_monotonic",
        "src_hyst_ladder_compression",
    ]
    for c in needed:
        _ensure(df, c)

    age_cap = config["hyst"]["age_cap_sec"]
    slope_cap = config["hyst"]["spread_slope_cap"]
    accel_cap = config["hyst"]["spread_accel_cap"]

    df["v21_hyst_primary_sign_num"] = df["src_hyst_primary_sign"].map(sign3)
    df["v21_hyst_probe_sign_num"] = df["src_hyst_probe_sign"].map(sign3)
    df["v21_hyst_episode_age_norm"] = df["src_hyst_episode_age_sec"].map(
        lambda x: clip01(safe_div(x, age_cap))
    )
    df["v21_hyst_last_cross_age_norm"] = df["src_hyst_last_cross_age_sec"].map(
        lambda x: clip01(safe_div(x, age_cap))
    )
    df["v21_hyst_probe_flip_watch"] = _num(df["src_hyst_probe_flip_watch"])
    df["v21_hyst_fast_collapse"] = _num(df["src_hyst_fast_collapse"])

    for side in ["s0", "s1"]:
        df[f"v21_hyst_{side}_state_norm"] = df[f"src_hyst_{side}_state"].map(
            lambda x: SPREAD_STATE_MAP.get(str(x).lower(), float("nan"))
        )
        df[f"v21_hyst_{side}_pressure_norm"] = df[f"src_hyst_{side}_pressure"].map(
            lambda x: PRESSURE_MAP.get(str(x).lower(), float("nan"))
        )
        df[f"v21_hyst_{side}_risk_norm"] = df[f"src_hyst_{side}_risk"].map(
            lambda x: STATE_RISK_MAP.get(str(x).lower(), float("nan"))
        )
        df[f"v21_hyst_{side}_stability_norm"] = _num(df[f"src_hyst_{side}_stability"])
        df[f"v21_hyst_{side}_near_cross"] = _num(df[f"src_hyst_{side}_near_cross"])
        df[f"v21_hyst_{side}_spread_slope_norm"] = df[
            f"src_hyst_{side}_spread_slope"
        ].map(lambda x: clip11(safe_div(x, slope_cap)))
        df[f"v21_hyst_{side}_spread_accel_norm"] = df[
            f"src_hyst_{side}_spread_accel"
        ].map(lambda x: clip11(safe_div(x, accel_cap)))

    df["v21_hyst_leader_max_sigma_norm"] = df["src_hyst_leader_max_sigma"].map(
        lambda x: clip01(safe_div(x, 83.0))
    )
    df["v21_hyst_leader_min_sigma_norm"] = df["src_hyst_leader_min_sigma"].map(
        lambda x: clip01(safe_div(x, 83.0))
    )
    df["v21_hyst_ladder_monotonic"] = _num(df["src_hyst_ladder_monotonic"])
    df["v21_hyst_ladder_compression"] = _num(df["src_hyst_ladder_compression"])
    df["v21_hyst_primary_agree_up"] = (df["v21_hyst_primary_sign_num"] > 0).astype(float)
    df["v21_hyst_primary_agree_down"] = (df["v21_hyst_primary_sign_num"] < 0).astype(float)

    def _cont_quality(r: pd.Series) -> float:
        stable_spread = 1.0 - mean_safe(
            [r["v21_hyst_s0_state_norm"], r["v21_hyst_s1_state_norm"]]
        )
        calm_pressure = 1.0 - mean_safe(
            [r["v21_hyst_s0_pressure_norm"], r["v21_hyst_s1_pressure_norm"]]
        )
        low_risk = 1.0 - mean_safe(
            [r["v21_hyst_s0_risk_norm"], r["v21_hyst_s1_risk_norm"]]
        )
        no_flip_watch = 1.0 - bool01(r["src_hyst_probe_flip_watch"])
        no_fast_collapse = 1.0 - bool01(r["src_hyst_fast_collapse"])
        ladder_coherence = mean_safe(
            [bool01(r["src_hyst_ladder_monotonic"]), 1.0 - bool01(r["src_hyst_ladder_compression"])]
        )
        return clip01(
            0.22 * stable_spread
            + 0.18 * calm_pressure
            + 0.18 * low_risk
            + 0.17 * no_flip_watch
            + 0.12 * no_fast_collapse
            + 0.13 * ladder_coherence
        )

    def _frozen_exhaustion(r: pd.Series) -> float:
        old_episode = r["v21_hyst_episode_age_norm"]
        frozen_state = mean_safe(
            [
                float(str(r["src_hyst_s0_state"]).lower() in ["stable", "frozen"]),
                float(str(r["src_hyst_s1_state"]).lower() in ["stable", "frozen"]),
            ]
        )
        calm_pressure = 1.0 - mean_safe(
            [r["v21_hyst_s0_pressure_norm"], r["v21_hyst_s1_pressure_norm"]]
        )
        no_probe_rescue = float(
            r["v21_hyst_probe_sign_num"] == 0
            or r["v21_hyst_probe_sign_num"] != r["v21_hyst_primary_sign_num"]
        )
        no_rewidening = mean_safe(
            [
                float(r["v21_hyst_s0_spread_accel_norm"] <= 0),
                float(r["v21_hyst_s1_spread_accel_norm"] <= 0),
            ]
        )
        no_near_cross = 1.0 - mean_safe(
            [bool01(r["src_hyst_s0_near_cross"]), bool01(r["src_hyst_s1_near_cross"])]
        )
        return clip01(
            0.25 * old_episode
            + 0.20 * frozen_state
            + 0.15 * calm_pressure
            + 0.15 * no_probe_rescue
            + 0.15 * no_rewidening
            + 0.10 * no_near_cross
        )

    def _instability(r: pd.Series) -> float:
        adverse_spread_accel = mean_safe(
            [
                float(r["v21_hyst_s0_spread_accel_norm"] < 0),
                float(r["v21_hyst_s1_spread_accel_norm"] < 0),
            ]
        )
        return clip01(
            0.30 * bool01(r["src_hyst_probe_flip_watch"])
            + 0.25 * bool01(r["src_hyst_fast_collapse"])
            + 0.20 * adverse_spread_accel
            + 0.15 * bool01(r["src_hyst_ladder_compression"])
            + 0.10 * mean_safe(
                [bool01(r["src_hyst_s0_near_cross"]), bool01(r["src_hyst_s1_near_cross"])]
            )
        )

    df["v21_hyst_continuation_quality"] = df.apply(_cont_quality, axis=1)
    df["v21_hyst_frozen_exhaustion"] = df.apply(_frozen_exhaustion, axis=1)
    df["v21_hyst_instability_risk"] = df.apply(_instability, axis=1)
    return df