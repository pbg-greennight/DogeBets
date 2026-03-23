from __future__ import annotations

import pandas as pd

from .common_v21 import SIGMAS_ALL, SIGMAS_FAST, SIGMAS_SLOW, clip01, mean_safe, safe_div


def _ensure(df: pd.DataFrame, col: str, default=float("nan")) -> None:
    if col not in df.columns:
        df[col] = default


def add_csd_dcsd_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    df["v21_src_missing_csd_dcsd"] = 0.0

    for sigma in SIGMAS_ALL:
        for c in [f"src_dcsd_l2_dmid_mean_s{sigma}", f"src_dcsd_l2_dwidth_mean_s{sigma}", f"src_dcsd_l2_dmid_flip_count_s{sigma}", f"src_dcsd_l2_dwidth_flip_count_s{sigma}"]:
            _ensure(df, c)

    flip_cap = config["csd_dcsd"]["flip_count_cap"]
    df["v21_dcsd_l2_fast_mid_slope_mean"] = df.apply(lambda r: mean_safe([r[f"src_dcsd_l2_dmid_mean_s{s}"] for s in SIGMAS_FAST]), axis=1)
    df["v21_dcsd_l2_slow_mid_slope_mean"] = df.apply(lambda r: mean_safe([r[f"src_dcsd_l2_dmid_mean_s{s}"] for s in SIGMAS_SLOW]), axis=1)
    df["v21_dcsd_l2_fast_width_change_mean"] = df.apply(lambda r: mean_safe([r[f"src_dcsd_l2_dwidth_mean_s{s}"] for s in SIGMAS_FAST]), axis=1)
    df["v21_dcsd_l2_slow_width_change_mean"] = df.apply(lambda r: mean_safe([r[f"src_dcsd_l2_dwidth_mean_s{s}"] for s in SIGMAS_SLOW]), axis=1)
    df["v21_dcsd_l2_fast_mid_flip_density"] = df.apply(lambda r: clip01(safe_div(mean_safe([r[f"src_dcsd_l2_dmid_flip_count_s{s}"] for s in SIGMAS_FAST]), flip_cap)), axis=1)
    df["v21_dcsd_l2_slow_mid_flip_density"] = df.apply(lambda r: clip01(safe_div(mean_safe([r[f"src_dcsd_l2_dmid_flip_count_s{s}"] for s in SIGMAS_SLOW]), flip_cap)), axis=1)
    df["v21_dcsd_l2_fast_width_flip_density"] = df.apply(lambda r: clip01(safe_div(mean_safe([r[f"src_dcsd_l2_dwidth_flip_count_s{s}"] for s in SIGMAS_FAST]), flip_cap)), axis=1)
    df["v21_dcsd_l2_slow_width_flip_density"] = df.apply(lambda r: clip01(safe_div(mean_safe([r[f"src_dcsd_l2_dwidth_flip_count_s{s}"] for s in SIGMAS_SLOW]), flip_cap)), axis=1)

    def _motion_bull(r: pd.Series) -> float:
        positive_mid = clip01(safe_div((r["v21_dcsd_l2_fast_mid_slope_mean"] + r["v21_dcsd_l2_slow_mid_slope_mean"]), config["csd_dcsd"]["mid_slope_cap"]))
        supportive_width = float(r["v21_dcsd_l2_fast_width_change_mean"] >= config["csd_dcsd"]["width_support_min"])
        low_flip = 1.0 - mean_safe([r["v21_dcsd_l2_fast_mid_flip_density"], r["v21_dcsd_l2_slow_mid_flip_density"], r["v21_dcsd_l2_fast_width_flip_density"], r["v21_dcsd_l2_slow_width_flip_density"]])
        return clip01(0.40 * positive_mid + 0.30 * supportive_width + 0.30 * low_flip)

    def _motion_bear(r: pd.Series) -> float:
        negative_mid = clip01(safe_div((-(r["v21_dcsd_l2_fast_mid_slope_mean"] + r["v21_dcsd_l2_slow_mid_slope_mean"])), config["csd_dcsd"]["mid_slope_cap"]))
        supportive_width = float(r["v21_dcsd_l2_fast_width_change_mean"] >= config["csd_dcsd"]["width_support_min"])
        low_flip = 1.0 - mean_safe([r["v21_dcsd_l2_fast_mid_flip_density"], r["v21_dcsd_l2_slow_mid_flip_density"], r["v21_dcsd_l2_fast_width_flip_density"], r["v21_dcsd_l2_slow_width_flip_density"]])
        return clip01(0.40 * negative_mid + 0.30 * supportive_width + 0.30 * low_flip)

    def _fast_osc(r: pd.Series) -> float:
        fast_slow_flip_gap = clip01((r["v21_dcsd_l2_fast_mid_flip_density"] - r["v21_dcsd_l2_slow_mid_flip_density"] + 1.0) / 2.0)
        amp_vs_anchor = clip01(safe_div(abs(r["v21_dcsd_l2_fast_mid_slope_mean"] - r["v21_dcsd_l2_slow_mid_slope_mean"]), config["csd_dcsd"]["mid_slope_gap_cap"]))
        return clip01(0.40 * r["v21_dcsd_l2_fast_mid_flip_density"] + 0.25 * r["v21_dcsd_l2_fast_width_flip_density"] + 0.20 * fast_slow_flip_gap + 0.15 * amp_vs_anchor)

    def _reexpand_fail(r: pd.Series) -> float:
        fast_expand_attempt = float(r["v21_dcsd_l2_fast_width_change_mean"] > config["csd_dcsd"]["fast_expand_thresh"])
        slow_still_compress = float(r["v21_dcsd_l2_slow_width_change_mean"] < config["csd_dcsd"]["slow_compress_thresh"])
        no_mid_confirm = float(abs(r["v21_dcsd_l2_slow_mid_slope_mean"]) < config["csd_dcsd"]["slow_mid_confirm_thresh"])
        short_lived_expand = float(r["v21_dcsd_l2_fast_width_flip_density"] > config["csd_dcsd"]["expand_flip_thresh"])
        return clip01(0.35 * fast_expand_attempt + 0.30 * slow_still_compress + 0.20 * no_mid_confirm + 0.15 * short_lived_expand)

    def _compress_persist(r: pd.Series) -> float:
        slow_width_compress = clip01(safe_div((-r["v21_dcsd_l2_slow_width_change_mean"]), config["csd_dcsd"]["width_compress_cap"]))
        all_sigma_compress_share = mean_safe([float(r[f"src_dcsd_l2_dwidth_mean_s{s}"] < 0) for s in SIGMAS_ALL])
        no_broadening = float(r["v21_dcsd_l2_fast_width_change_mean"] <= 0 and r["v21_dcsd_l2_slow_width_change_mean"] <= 0)
        return clip01(0.50 * slow_width_compress + 0.30 * all_sigma_compress_share + 0.20 * no_broadening)

    df["v21_motion_trend_quality_bull"] = df.apply(_motion_bull, axis=1)
    df["v21_motion_trend_quality_bear"] = df.apply(_motion_bear, axis=1)
    df["v21_channel_fast_oscillation"] = df.apply(_fast_osc, axis=1)
    df["v21_channel_reexpansion_failure"] = df.apply(_reexpand_fail, axis=1)
    df["v21_channel_compression_persistence"] = df.apply(_compress_persist, axis=1)
    return df
