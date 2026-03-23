from __future__ import annotations

import pandas as pd

from .common_v21 import (
    SIGMAS_ALL,
    SIGMAS_CORE,
    SIGMAS_FAST,
    SIGMAS_SLOW,
    TRANSFER_DIR_MAP,
    TRANSFER_STATE_MAP,
    clip01,
    mean_safe,
    safe_div,
    sign3,
)


def _ensure_columns(df: pd.DataFrame, defaults: dict[str, object]) -> pd.DataFrame:
    """
    Ensure required source columns exist without fragmenting the DataFrame.
    """
    missing = {col: default for col, default in defaults.items() if col not in df.columns}
    if not missing:
        return df
    return pd.concat([df, pd.DataFrame(missing, index=df.index)], axis=1)


def add_msbc_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()

    required_defaults: dict[str, object] = {"v21_src_missing_msbc": 0.0}

    for sigma in SIGMAS_ALL:
        required_defaults.update({
            f"src_msbc_l1_slope_s{sigma}": float("nan"),
            f"src_msbc_l2_slope_s{sigma}": float("nan"),
            f"src_msbc_l1_curve_s{sigma}": float("nan"),
            f"src_msbc_l2_curve_s{sigma}": float("nan"),
            f"src_msbc_l2_lin_r2_s{sigma}": float("nan"),
            f"src_msbc_l2_quad_tangent_s{sigma}": float("nan"),
            f"src_msbc_l2_quad_curv_s{sigma}": float("nan"),
            f"src_msbc_l2_z_s{sigma}": float("nan"),
            f"src_msbc_l2_run_score_s{sigma}": float("nan"),
            f"src_msbc_override_s{sigma}": float("nan"),
        })

    required_defaults.update({
        "src_msbc_transfer_dir": "none",
        "src_msbc_transfer_depth": 0.0,
        "src_msbc_transfer_state": "none",
        "src_msbc_disorder_count": 0.0,
        "src_msbc_override_count": 0.0,
        "src_msbc_propagation_agree_ratio": 0.0,
        "src_msbc_propagation_disagree_ratio": 0.0,
    })

    df = _ensure_columns(df, required_defaults)

    new_cols: dict[str, object] = {}
    depth_max = config["msbc"]["transfer_depth_max"]
    disorder_max = config["msbc"]["disorder_max"]

    # Per-sigma columns
    for sigma in SIGMAS_ALL:
        l1_slope = df[f"src_msbc_l1_slope_s{sigma}"]
        l2_slope = df[f"src_msbc_l2_slope_s{sigma}"]
        l1_curve = df[f"src_msbc_l1_curve_s{sigma}"]
        l2_curve = df[f"src_msbc_l2_curve_s{sigma}"]

        new_cols[f"v21_msbc_l2_slope_s{sigma}"] = l2_slope
        new_cols[f"v21_msbc_l2_curve_s{sigma}"] = l2_curve
        new_cols[f"v21_msbc_l2_lin_r2_s{sigma}"] = df[f"src_msbc_l2_lin_r2_s{sigma}"]
        new_cols[f"v21_msbc_l2_quad_tangent_s{sigma}"] = df[f"src_msbc_l2_quad_tangent_s{sigma}"]
        new_cols[f"v21_msbc_l2_quad_curv_s{sigma}"] = df[f"src_msbc_l2_quad_curv_s{sigma}"]
        new_cols[f"v21_msbc_l2_z_s{sigma}"] = df[f"src_msbc_l2_z_s{sigma}"]
        new_cols[f"v21_msbc_l2_run_score_s{sigma}"] = df[f"src_msbc_l2_run_score_s{sigma}"]
        new_cols[f"v21_msbc_l2_dir_s{sigma}"] = l2_slope.map(sign3)
        new_cols[f"v21_msbc_l1_l2_sign_same_s{sigma}"] = [
            float(sign3(a) == sign3(b) and sign3(b) != 0) for a, b in zip(l1_slope, l2_slope)
        ]
        new_cols[f"v21_msbc_l1_l2_curve_same_s{sigma}"] = [
            float(sign3(a) == sign3(b) and sign3(b) != 0) for a, b in zip(l1_curve, l2_curve)
        ]
        new_cols[f"v21_msbc_override_s{sigma}"] = df[f"src_msbc_override_s{sigma}"].fillna(0.0).astype(float)
        new_cols[f"v21_msbc_slope_ratio_s{sigma}"] = [
            clip01(safe_div(abs(b), abs(a) + 1e-12)) for a, b in zip(l1_slope, l2_slope)
        ]
        new_cols[f"v21_msbc_curve_ratio_s{sigma}"] = [
            clip01(safe_div(abs(b), abs(a) + 1e-12)) for a, b in zip(l1_curve, l2_curve)
        ]

    # Stack-level columns
    new_cols["v21_msbc_transfer_dir_num"] = df["src_msbc_transfer_dir"].map(
        lambda x: TRANSFER_DIR_MAP.get(str(x).lower(), 0.0)
    )
    new_cols["v21_msbc_transfer_depth_norm"] = df["src_msbc_transfer_depth"].map(
        lambda x: clip01(safe_div(x, depth_max))
    )
    new_cols["v21_msbc_transfer_state_norm"] = df["src_msbc_transfer_state"].map(
        lambda x: TRANSFER_STATE_MAP.get(str(x).lower(), 0.0)
    )
    new_cols["v21_msbc_disorder_norm"] = df["src_msbc_disorder_count"].map(
        lambda x: clip01(safe_div(x, disorder_max))
    )
    new_cols["v21_msbc_override_count_norm"] = df["src_msbc_override_count"].map(
        lambda x: clip01(safe_div(x, 6.0))
    )

    # Build temporary frame for row-wise access to new cols
    tmp = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    new_cols["v21_msbc_fast_override_count_norm"] = tmp.apply(
        lambda r: clip01(mean_safe([r[f"v21_msbc_override_s{s}"] for s in SIGMAS_FAST])), axis=1
    )
    new_cols["v21_msbc_slow_override_count_norm"] = tmp.apply(
        lambda r: clip01(mean_safe([r[f"v21_msbc_override_s{s}"] for s in SIGMAS_SLOW])), axis=1
    )
    new_cols["v21_msbc_prop_agree_ratio"] = df["src_msbc_propagation_agree_ratio"].fillna(0.0)
    new_cols["v21_msbc_prop_disagree_ratio"] = df["src_msbc_propagation_disagree_ratio"].fillna(0.0)

    tmp = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    new_cols["v21_msbc_fast_dir_mean"] = tmp.apply(
        lambda r: mean_safe([r[f"v21_msbc_l2_dir_s{s}"] for s in SIGMAS_FAST]), axis=1
    )
    new_cols["v21_msbc_slow_dir_mean"] = tmp.apply(
        lambda r: mean_safe([r[f"v21_msbc_l2_dir_s{s}"] for s in SIGMAS_SLOW]), axis=1
    )

    tmp = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _fast_slow_conflict(r: pd.Series) -> float:
        dir_gap = abs(r["v21_msbc_fast_dir_mean"] - r["v21_msbc_slow_dir_mean"]) / 2.0
        mixed_core = 1.0 - abs(mean_safe([r[f"v21_msbc_l2_dir_s{s}"] for s in SIGMAS_CORE]))
        override_term = r["v21_msbc_fast_override_count_norm"]
        return clip01(0.50 * dir_gap + 0.25 * mixed_core + 0.25 * override_term)

    new_cols["v21_msbc_fast_slow_conflict"] = tmp.apply(_fast_slow_conflict, axis=1)

    tmp = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _consistency_break(r: pd.Series) -> float:
        sign_same_mean = mean_safe([r[f"v21_msbc_l1_l2_sign_same_s{s}"] for s in SIGMAS_ALL])
        curve_same_mean = mean_safe([r[f"v21_msbc_l1_l2_curve_same_s{s}"] for s in SIGMAS_ALL])
        return clip01(1.0 - (0.60 * sign_same_mean + 0.40 * curve_same_mean))

    new_cols["v21_msbc_consistency_break"] = tmp.apply(_consistency_break, axis=1)

    tmp = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    new_cols["v21_msbc_fragility"] = (
        0.35 * tmp["v21_msbc_override_count_norm"]
        + 0.25 * tmp["v21_msbc_fast_override_count_norm"]
        + 0.25 * tmp["v21_msbc_fast_slow_conflict"]
        + 0.15 * tmp["v21_msbc_consistency_break"]
    ).map(clip01)

    tmp = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _bull_alignment(r: pd.Series) -> float:
        bull_dir_agree = mean_safe([float(r[f"v21_msbc_l2_slope_s{s}"] > 0) for s in SIGMAS_CORE])
        transfer_up = float(r["v21_msbc_transfer_dir_num"] > 0)
        low_disorder = 1.0 - r["v21_msbc_disorder_norm"]
        low_override = 1.0 - r["v21_msbc_override_count_norm"]
        return clip01(
            0.45 * bull_dir_agree
            + 0.25 * (transfer_up * r["v21_msbc_transfer_depth_norm"])
            + 0.15 * low_disorder
            + 0.15 * low_override
        )

    def _bear_alignment(r: pd.Series) -> float:
        bear_dir_agree = mean_safe([float(r[f"v21_msbc_l2_slope_s{s}"] < 0) for s in SIGMAS_CORE])
        transfer_down = float(r["v21_msbc_transfer_dir_num"] < 0)
        low_disorder = 1.0 - r["v21_msbc_disorder_norm"]
        low_override = 1.0 - r["v21_msbc_override_count_norm"]
        return clip01(
            0.45 * bear_dir_agree
            + 0.25 * (transfer_down * r["v21_msbc_transfer_depth_norm"])
            + 0.15 * low_disorder
            + 0.15 * low_override
        )

    def _run_maturity(r: pd.Series) -> float:
        ref_dir = sign3(r["v21_msbc_l2_slope_s53"])
        slow_align = mean_safe(
            [float(sign3(r[f"v21_msbc_l2_slope_s{s}"]) == ref_dir and ref_dir != 0) for s in SIGMAS_SLOW]
        )
        fast_ratio_fade = 1.0 - mean_safe([r["v21_msbc_slope_ratio_s8"], r["v21_msbc_slope_ratio_s23"]])
        slow_flatten = mean_safe([
            clip01(1.0 - abs(r["v21_msbc_l2_quad_curv_s53"])),
            clip01(1.0 - abs(r["v21_msbc_l2_quad_curv_s68"])),
            clip01(1.0 - abs(r["v21_msbc_l2_quad_curv_s83"])),
        ])
        return clip01(0.45 * slow_align + 0.30 * fast_ratio_fade + 0.25 * slow_flatten)

    new_cols["v21_msbc_bull_alignment"] = tmp.apply(_bull_alignment, axis=1)
    new_cols["v21_msbc_bear_alignment"] = tmp.apply(_bear_alignment, axis=1)
    new_cols["v21_msbc_run_maturity"] = tmp.apply(_run_maturity, axis=1)

    msbc_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, msbc_df], axis=1)
