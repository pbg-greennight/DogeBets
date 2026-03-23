from __future__ import annotations

import numpy as np
import pandas as pd

from .segment_engine import build_segment_feature_table


EPS = 1e-12


def _safe_div(a, b):
    return a / (b + EPS)


def _gaussian_kernel_1d(sigma: float, window_factor: int = 4) -> np.ndarray:
    sigma = float(sigma)
    radius = max(1, int(round(window_factor * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def _causal_gaussian_smooth(series: pd.Series, sigma: float, window_factor: int = 4) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    kernel = _gaussian_kernel_1d(sigma=sigma, window_factor=window_factor)

    center = len(kernel) // 2
    causal_kernel = kernel[: center + 1].copy()
    causal_kernel = causal_kernel[::-1]

    out = np.full_like(values, fill_value=np.nan, dtype=float)

    for i in range(len(values)):
        start = max(0, i - len(causal_kernel) + 1)
        hist = values[start : i + 1]

        k = causal_kernel[: len(hist)].copy()

        valid_mask = np.isfinite(hist)
        if not valid_mask.any():
            continue

        hist_valid = hist[valid_mask]
        k_valid = k[valid_mask]

        denom = k_valid.sum()
        if denom <= 0:
            continue

        out[i] = np.dot(hist_valid, k_valid) / denom

    return pd.Series(out, index=series.index)


def add_base_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = out["close"].pct_change()
    out["logret_1"] = np.log(out["close"] / out["close"].shift(1))
    out["hl_range"] = _safe_div(out["high"] - out["low"], out["close"])
    out["body"] = _safe_div(out["close"] - out["open"], out["open"])

    out["close_loc"] = _safe_div(
        out["close"] - out["low"],
        (out["high"] - out["low"]).replace(0, np.nan)
    ).fillna(0.5)

    out["vwap_dist"] = _safe_div(out["close"] - out["vwap"], out["close"])
    out["vol_chg"] = out["volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    if "bars_used" in out.columns and "bars_total" in out.columns:
        out["bars_used_ratio"] = _safe_div(out["bars_used"], out["bars_total"])
    if "seconds_to_boundary_from_last_bar" in out.columns:
        out["boundary_gap_norm"] = _safe_div(out["seconds_to_boundary_from_last_bar"], 300.0)
    if "tail_ret_3" in out.columns and "tail_ret_5" in out.columns:
        out["tail_ret_accel_3_5"] = out["tail_ret_3"] - out["tail_ret_5"]
    if "tail_ret_5" in out.columns and "tail_ret_10" in out.columns:
        out["tail_ret_accel_5_10"] = out["tail_ret_5"] - out["tail_ret_10"]

    fe_cfg = config["feature_engineering"]

    for w in fe_cfg.get("trend_windows", [3, 5, 8, 13, 21]):
        out[f"mom_{w}"] = out["close"].pct_change(w)
        out[f"ema_{w}"] = out["close"].ewm(span=w, adjust=False).mean()
        out[f"ema_slope_{w}"] = out[f"ema_{w}"].diff()
        out[f"ema_curv_{w}"] = out[f"ema_slope_{w}"].diff()

    for w in fe_cfg.get("vol_windows", [5, 10, 20]):
        out[f"rv_{w}"] = out["logret_1"].rolling(w).std()
        out[f"ret_z_{w}"] = _safe_div(out["ret_1"], out[f"rv_{w}"])

    for w in fe_cfg.get("pressure_windows", [3, 5, 10]):
        out[f"body_mean_{w}"] = out["body"].rolling(w).mean()
        out[f"close_loc_mean_{w}"] = out["close_loc"].rolling(w).mean()
        out[f"vol_mean_{w}"] = out["volume"].rolling(w).mean()
        out[f"vol_z_{w}"] = _safe_div(
            out["volume"] - out[f"vol_mean_{w}"],
            out["volume"].rolling(w).std()
        )

    for w in fe_cfg.get("regime_windows", [10, 20, 40]):
        out[f"trendiness_{w}"] = _safe_div(
            out["close"].diff(w).abs(),
            out["close"].diff().abs().rolling(w).sum()
        )

        out[f"range_pos_{w}"] = _safe_div(
            out["close"] - out["close"].rolling(w).min(),
            out["close"].rolling(w).max() - out["close"].rolling(w).min()
        )

        base_denom = (
            out["rv_5"].rolling(w).mean()
            if "rv_5" in out.columns
            else out["hl_range"].rolling(w).std()
        )
        out[f"compression_{w}"] = _safe_div(out["hl_range"].rolling(w).mean(), base_denom)

    out["hour"] = pd.to_datetime(out["next_epoch_time"]).dt.hour
    out["minute"] = pd.to_datetime(out["next_epoch_time"]).dt.minute
    out["epoch_mod_2"] = out["epoch"] % 2
    out["epoch_mod_3"] = out["epoch"] % 3

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def add_gaussian_stack_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()

    gs_cfg = config.get("feature_engineering", {}).get("gaussian_stack", {})
    enabled = bool(gs_cfg.get("enabled", False))
    if not enabled:
        return out

    sigmas = gs_cfg.get("sigmas", [2, 8, 23, 38, 53, 68, 83])
    window_factor = int(gs_cfg.get("window_factor", 4))

    close = pd.to_numeric(out["close"], errors="coerce").astype(float)

    g_cols = []
    g_slope_cols = []
    g_curv_cols = []

    for sigma in sigmas:
        sigma_tag = str(sigma).replace(".", "_")

        g_col = f"g_{sigma_tag}"
        g_slope_col = f"g_slope_{sigma_tag}"
        g_curv_col = f"g_curv_{sigma_tag}"
        g_dist_col = f"g_dist_{sigma_tag}"
        g_slope_norm_col = f"g_slope_norm_{sigma_tag}"
        g_curv_norm_col = f"g_curv_norm_{sigma_tag}"

        out[g_col] = _causal_gaussian_smooth(close, sigma=float(sigma), window_factor=window_factor)
        out[g_slope_col] = out[g_col].diff()
        out[g_curv_col] = out[g_slope_col].diff()

        out[g_dist_col] = _safe_div(close - out[g_col], close.abs() + EPS)
        out[g_slope_norm_col] = _safe_div(out[g_slope_col], out[g_col].abs() + EPS)
        out[g_curv_norm_col] = _safe_div(out[g_curv_col], out[g_col].abs() + EPS)

        g_cols.append(g_col)
        g_slope_cols.append(g_slope_col)
        g_curv_cols.append(g_curv_col)

    if g_cols:
        g_matrix = out[g_cols]
        out["g_stack_max"] = g_matrix.max(axis=1)
        out["g_stack_min"] = g_matrix.min(axis=1)
        out["g_stack_spread"] = out["g_stack_max"] - out["g_stack_min"]
        out["g_stack_centroid"] = g_matrix.mean(axis=1)
        out["g_price_vs_centroid"] = _safe_div(close - out["g_stack_centroid"], close.abs() + EPS)

    if g_slope_cols:
        slope_matrix = out[g_slope_cols]
        out["g_slope_mean"] = slope_matrix.mean(axis=1)
        out["g_slope_std"] = slope_matrix.std(axis=1)
        out["g_slope_pos_count"] = (slope_matrix > 0).sum(axis=1)
        out["g_slope_neg_count"] = (slope_matrix < 0).sum(axis=1)
        out["g_slope_agreement"] = _safe_div(
            (slope_matrix > 0).sum(axis=1) - (slope_matrix < 0).sum(axis=1),
            len(g_slope_cols)
        )

    if g_curv_cols:
        curv_matrix = out[g_curv_cols]
        out["g_curv_mean"] = curv_matrix.mean(axis=1)
        out["g_curv_std"] = curv_matrix.std(axis=1)
        out["g_curv_pos_count"] = (curv_matrix > 0).sum(axis=1)
        out["g_curv_neg_count"] = (curv_matrix < 0).sum(axis=1)
        out["g_curv_agreement"] = _safe_div(
            (curv_matrix > 0).sum(axis=1) - (curv_matrix < 0).sum(axis=1),
            len(g_curv_cols)
        )

    for i in range(len(sigmas) - 1):
        s1 = str(sigmas[i]).replace(".", "_")
        s2 = str(sigmas[i + 1]).replace(".", "_")

        out[f"g_delta_{s1}_{s2}"] = out[f"g_{s1}"] - out[f"g_{s2}"]
        out[f"g_slope_delta_{s1}_{s2}"] = out[f"g_slope_{s1}"] - out[f"g_slope_{s2}"]
        out[f"g_curv_delta_{s1}_{s2}"] = out[f"g_curv_{s1}"] - out[f"g_curv_{s2}"]

    if len(sigmas) >= 2:
        short_tag = str(sigmas[0]).replace(".", "_")
        long_tag = str(sigmas[-1]).replace(".", "_")

        out["g_short_long_value_delta"] = out[f"g_{short_tag}"] - out[f"g_{long_tag}"]
        out["g_short_long_slope_delta"] = out[f"g_slope_{short_tag}"] - out[f"g_slope_{long_tag}"]
        out["g_short_long_curv_delta"] = out[f"g_curv_{short_tag}"] - out[f"g_curv_{long_tag}"]

        out["g_short_above_long"] = (out[f"g_{short_tag}"] > out[f"g_{long_tag}"]).astype(int)
        out["g_short_slope_above_long"] = (
            out[f"g_slope_{short_tag}"] > out[f"g_slope_{long_tag}"]
        ).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def add_fan_geometry_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()

    fg_cfg = config.get("feature_engineering", {}).get("fan_geometry", {})
    enabled = bool(fg_cfg.get("enabled", False))
    if not enabled:
        return out

    sigmas = config.get("feature_engineering", {}).get("gaussian_stack", {}).get(
        "sigmas", [2, 8, 23, 38, 53, 68, 83]
    )
    sigma_tags = [str(s).replace(".", "_") for s in sigmas]

    g_cols = [f"g_{s}" for s in sigma_tags if f"g_{s}" in out.columns]
    g_slope_cols = [f"g_slope_{s}" for s in sigma_tags if f"g_slope_{s}" in out.columns]
    g_curv_cols = [f"g_curv_{s}" for s in sigma_tags if f"g_curv_{s}" in out.columns]

    if len(g_cols) < 2:
        return out

    g_matrix = out[g_cols]
    slope_matrix = out[g_slope_cols] if g_slope_cols else None
    curv_matrix = out[g_curv_cols] if g_curv_cols else None

    short_g = g_cols[0]
    long_g = g_cols[-1]

    out["fan_width"] = out[short_g] - out[long_g]
    out["fan_abs_width"] = out["fan_width"].abs()
    out["fan_width_velocity"] = out["fan_width"].diff()
    out["fan_width_acceleration"] = out["fan_width_velocity"].diff()

    out["outer_fan_width"] = out[g_cols[1]] - out[g_cols[-2]] if len(g_cols) >= 4 else out["fan_width"]
    out["outer_fan_abs_width"] = out["outer_fan_width"].abs()
    out["outer_fan_width_velocity"] = out["outer_fan_width"].diff()

    out["fan_internal_spread"] = g_matrix.max(axis=1) - g_matrix.min(axis=1)
    out["fan_internal_spread_velocity"] = out["fan_internal_spread"].diff()
    out["fan_internal_spread_acceleration"] = out["fan_internal_spread_velocity"].diff()

    desc_score_parts = []
    asc_score_parts = []
    violation_count = np.zeros(len(out), dtype=float)

    for i in range(len(g_cols) - 1):
        left = out[g_cols[i]]
        right = out[g_cols[i + 1]]

        desc_ok = (left >= right).astype(float)
        asc_ok = (left <= right).astype(float)

        desc_score_parts.append(desc_ok)
        asc_score_parts.append(asc_ok)
        violation_count += (left < right).astype(float)

    desc_score = np.mean(np.vstack(desc_score_parts), axis=0)
    asc_score = np.mean(np.vstack(asc_score_parts), axis=0)

    out["fan_order_score"] = desc_score - asc_score
    out["fan_order_desc_score"] = desc_score
    out["fan_order_asc_score"] = asc_score
    out["fan_order_violation_count"] = violation_count
    out["fan_order_violation_ratio"] = _safe_div(violation_count, max(len(g_cols) - 1, 1))

    out["fan_order_score_velocity"] = out["fan_order_score"].diff()
    out["fan_order_score_acceleration"] = out["fan_order_score_velocity"].diff()
    out["fan_order_flip_flag"] = (
        (out["fan_order_score"].shift(1) * out["fan_order_score"]) < 0
    ).astype(int)

    out["fan_short_above_long"] = (out[short_g] > out[long_g]).astype(int)
    out["fan_short_below_long"] = (out[short_g] < out[long_g]).astype(int)

    if len(g_cols) >= 3:
        mid_g = g_cols[len(g_cols) // 2]
        out["fan_short_above_mid"] = (out[short_g] > out[mid_g]).astype(int)
        out["fan_mid_above_long"] = (out[mid_g] > out[long_g]).astype(int)
        out["fan_hierarchy_consistent"] = (
            (out["fan_short_above_mid"] == 1) & (out["fan_mid_above_long"] == 1)
        ).astype(int)

    out["fan_is_expanding"] = (out["fan_width_velocity"] > 0).astype(int)
    out["fan_is_compressing"] = (out["fan_width_velocity"] < 0).astype(int)
    out["fan_compression_ratio"] = _safe_div(out["fan_width_velocity"], out["fan_abs_width"] + EPS)
    out["fan_expansion_strength"] = _safe_div(out["fan_internal_spread_velocity"], out["fan_internal_spread"].abs() + EPS)

    out["fan_centroid"] = g_matrix.mean(axis=1)
    out["fan_price_vs_centroid"] = _safe_div(out["close"] - out["fan_centroid"], out["close"].abs() + EPS)

    upper_dist = g_matrix.sub(out["fan_centroid"], axis=0).clip(lower=0).sum(axis=1)
    lower_dist = (-g_matrix.sub(out["fan_centroid"], axis=0).clip(upper=0)).sum(axis=1)
    out["fan_skew"] = _safe_div(upper_dist - lower_dist, upper_dist + lower_dist + EPS)

    if slope_matrix is not None and len(g_slope_cols) >= 2:
        out["fan_slope_span"] = slope_matrix.max(axis=1) - slope_matrix.min(axis=1)
        out["fan_slope_span_velocity"] = out["fan_slope_span"].diff()

        short_slope = g_slope_cols[0]
        long_slope = g_slope_cols[-1]

        out["fan_slope_short_long_delta"] = out[short_slope] - out[long_slope]
        out["fan_slope_short_long_abs_delta"] = out["fan_slope_short_long_delta"].abs()
        out["fan_slope_alignment_strength"] = _safe_div(
            slope_matrix.mean(axis=1),
            slope_matrix.std(axis=1) + EPS
        )

    if curv_matrix is not None and len(g_curv_cols) >= 2:
        out["fan_curv_span"] = curv_matrix.max(axis=1) - curv_matrix.min(axis=1)
        out["fan_curv_span_velocity"] = out["fan_curv_span"].diff()

        short_curv = g_curv_cols[0]
        long_curv = g_curv_cols[-1]

        out["fan_curv_short_long_delta"] = out[short_curv] - out[long_curv]
        out["fan_curv_short_long_abs_delta"] = out["fan_curv_short_long_delta"].abs()
        out["g_long_curvature"] = out[long_curv]

    if "rv_5" in out.columns:
        out["fan_width_vol_norm"] = _safe_div(out["fan_width"], out["rv_5"] + EPS)
        out["outer_fan_width_vol_norm"] = _safe_div(out["outer_fan_width"], out["rv_5"] + EPS)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def add_segment_engine_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()

    se_cfg = config.get("feature_engineering", {}).get("segment_engine", {})
    enabled = bool(se_cfg.get("enabled", False))
    if not enabled:
        return out

    g83_col = se_cfg.get("g83_col", "g_83")
    seg_df = build_segment_feature_table(out, g83_col=g83_col)

    # merge by aligned index
    out = pd.concat([out, seg_df], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_feature_matrix(df: pd.DataFrame, feature_blocks: list[str], config: dict) -> tuple[pd.DataFrame, list[str]]:
    df = add_base_features(df, config)

    if "gaussian_stack" in feature_blocks:
        df = add_gaussian_stack_features(df, config)

    if "fan_geometry" in feature_blocks:
        df = add_fan_geometry_features(df, config)

    if "segment_engine" in feature_blocks:
        df = add_segment_engine_features(df, config)

    feature_cols: list[str] = []

    if "trend" in feature_blocks:
        feature_cols += [
            c for c in df.columns
            if c.startswith(("mom_", "ema_", "ema_slope_", "ema_curv_"))
        ]

    if "pressure" in feature_blocks:
        feature_cols += [
            c for c in df.columns
            if c.startswith(("body_mean_", "close_loc_mean_", "vol_mean_", "vol_z_"))
        ]
        feature_cols += ["body", "close_loc", "vwap_dist", "vol_chg"]

        for c in ["tail_ret_3", "tail_ret_5", "tail_ret_10", "tail_ret_accel_3_5", "tail_ret_accel_5_10"]:
            if c in df.columns:
                feature_cols.append(c)

    if "regime" in feature_blocks:
        feature_cols += [
            c for c in df.columns
            if c.startswith(("trendiness_", "range_pos_", "compression_"))
        ]

        for c in ["bars_used_ratio", "boundary_gap_norm", "coverage_ratio"]:
            if c in df.columns:
                feature_cols.append(c)

    if "volatility" in feature_blocks:
        feature_cols += [
            c for c in df.columns
            if c.startswith(("rv_", "ret_z_"))
        ]
        feature_cols += ["hl_range"]

    if "time" in feature_blocks:
        feature_cols += ["hour", "minute", "epoch_mod_2", "epoch_mod_3"]

    if "gaussian_stack" in feature_blocks:
        feature_cols += [
            c for c in df.columns
            if c.startswith((
                "g_",
                "g_slope_",
                "g_curv_",
                "g_dist_",
                "g_slope_norm_",
                "g_curv_norm_",
                "g_stack_",
                "g_price_vs_centroid",
                "g_delta_",
                "g_slope_delta_",
                "g_curv_delta_",
                "g_short_long_",
            ))
        ]

    if "fan_geometry" in feature_blocks:
        feature_cols += [
            c for c in df.columns
            if c.startswith((
                "fan_width",
                "fan_abs_width",
                "fan_internal_spread",
                "fan_order_",
                "fan_short_",
                "fan_mid_",
                "fan_hierarchy_",
                "fan_is_",
                "fan_compression_",
                "fan_expansion_",
                "fan_centroid",
                "fan_price_vs_centroid",
                "fan_skew",
                "fan_slope_",
                "fan_curv_",
                "fan_width_vol_norm",
                "outer_fan_",
            ))
        ]
        if "g_long_curvature" in df.columns:
            feature_cols.append("g_long_curvature")

    if "segment_engine" in feature_blocks:
        segment_cols = [
            "segment_valid",
            "segment_points",
            "gap_ratio",
            "missing_sum",
            "anchor_epoch",
            "anchor_is_peak",
            "anchor_is_valley",
            "segment_len_epochs",
            "tail_len_epochs",
            "tail_method_adaptive",
            "tail_method_fallback",
            "segment_dir_up",
            "segment_dir_down",
            "segment_dir_flat",
            "g83_anchor_to_now_delta",
            "seg_g83_slope",
            "seg_g83_r2",
            "seg_g83_quad_tangent",
            "seg_g83_quad_curv",
            "tail_g83_slope",
            "tail_g83_r2",
            "tail_g83_quad_tangent",
            "tail_g83_quad_curv",
            "seg_energy_total",
            "seg_momentum_score",
            "tail_energy_total",
            "tail_momentum_score",
            "seg_plateau_score",
            "tail_plateau_score",
            "tail_fast_slow_slope_delta",
            "tail_fast_slow_curv_delta",
            "fan_spread_now",
            "fan_spread_mean",
            "fan_spread_slope",
            "fan_spread_accel",
            "fan_inversion_count",
            "fan_order_violation_now",
            "tail_fan_spread_now",
            "tail_fan_spread_slope",
            "tail_fan_spread_accel",
            "tail_fan_inversion_count",
            "tail_fan_order_violation_now",
            "price_anchor_to_now_delta",
            "tail_price_delta",
        ]
        for c in segment_cols:
            if c in df.columns:
                feature_cols.append(c)

    feature_cols = sorted(set(feature_cols))
    x = df[feature_cols].copy()
    x = x.ffill().bfill().fillna(0.0)

    return x, feature_cols