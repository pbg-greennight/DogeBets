from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class AnchorInfo:
    anchor_idx: int
    anchor_epoch: int
    anchor_type: str   # PEAK / VALLEY


@dataclass
class TailInfo:
    tail_start_idx: int
    tail_method: str   # adaptive_extremum / fallback_fraction


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _is_local_peak(vals: np.ndarray, i: int) -> bool:
    if i <= 0 or i >= len(vals) - 1:
        return False
    a, b, c = vals[i - 1], vals[i], vals[i + 1]
    return np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and (a < b > c)


def _is_local_valley(vals: np.ndarray, i: int) -> bool:
    if i <= 0 or i >= len(vals) - 1:
        return False
    a, b, c = vals[i - 1], vals[i], vals[i + 1]
    return np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and (a > b < c)


def _linreg_metrics(y: np.ndarray) -> dict:
    n = len(y)
    if n < 2:
        return {
            "lin_slope": 0.0,
            "lin_r2": 0.0,
            "quad_a": 0.0,
            "quad_b": 0.0,
            "quad_r2": 0.0,
            "quad_tangent": 0.0,
            "quad_curv": 0.0,
        }

    x = np.arange(n, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(y)
    if n < 2:
        return {
            "lin_slope": 0.0,
            "lin_r2": 0.0,
            "quad_a": 0.0,
            "quad_b": 0.0,
            "quad_r2": 0.0,
            "quad_tangent": 0.0,
            "quad_curv": 0.0,
        }

    # linear
    p1 = np.polyfit(x, y, 1)
    yhat1 = np.polyval(p1, x)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res1 = float(np.sum((y - yhat1) ** 2))
    lin_r2 = 0.0 if ss_tot <= EPS else max(0.0, 1.0 - ss_res1 / ss_tot)

    out = {
        "lin_slope": float(p1[0]),
        "lin_r2": float(lin_r2),
        "quad_a": 0.0,
        "quad_b": 0.0,
        "quad_r2": 0.0,
        "quad_tangent": 0.0,
        "quad_curv": 0.0,
    }

    # quadratic
    if n >= 3 and len(np.unique(x)) >= 3:
        p2 = np.polyfit(x, y, 2)
        yhat2 = np.polyval(p2, x)
        ss_res2 = float(np.sum((y - yhat2) ** 2))
        quad_r2 = 0.0 if ss_tot <= EPS else max(0.0, 1.0 - ss_res2 / ss_tot)

        x_last = x[-1]
        quad_tangent = float(2.0 * p2[0] * x_last + p2[1])
        quad_curv = float(2.0 * p2[0])

        out.update({
            "quad_a": float(p2[0]),
            "quad_b": float(p2[1]),
            "quad_r2": float(quad_r2),
            "quad_tangent": quad_tangent,
            "quad_curv": quad_curv,
        })

    return out


def _segment_energy(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return {
            "energy_total": 0.0,
            "energy_mean": 0.0,
            "energy_std": 0.0,
            "momentum_score": 0.0,
        }

    dy = np.diff(y)
    energy_total = float(np.sum(np.abs(dy)))
    energy_mean = float(np.mean(np.abs(dy)))
    energy_std = float(np.std(dy))

    net_move = float(y[-1] - y[0])
    momentum_score = 0.0 if energy_total <= EPS else float(abs(net_move) / energy_total)

    return {
        "energy_total": energy_total,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "momentum_score": momentum_score,
    }


def _plateau_score(y: np.ndarray, flat_quantile: float = 0.25) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) < 4:
        return 0.0

    dy = np.abs(np.diff(y))
    if len(dy) == 0:
        return 0.0

    thresh = np.quantile(dy, flat_quantile)
    if thresh <= EPS:
        thresh = EPS

    frac_flat = float(np.mean(dy <= thresh))
    return frac_flat


def _fan_spread_features(df_seg: pd.DataFrame, sigma_cols: list[str]) -> dict:
    valid_cols = [c for c in sigma_cols if c in df_seg.columns]
    if len(valid_cols) < 2:
        return {
            "fan_spread_now": 0.0,
            "fan_spread_mean": 0.0,
            "fan_spread_slope": 0.0,
            "fan_spread_accel": 0.0,
            "fan_inversion_count": 0,
            "fan_order_violation_now": 0,
        }

    mat = df_seg[valid_cols].astype(float).to_numpy()
    spread = np.nanmax(mat, axis=1) - np.nanmin(mat, axis=1)

    spread_reg = _linreg_metrics(spread)

    # order violations at current row
    last = mat[-1]
    viol_now = 0
    for i in range(len(last) - 1):
        if np.isfinite(last[i]) and np.isfinite(last[i + 1]) and last[i] < last[i + 1]:
            viol_now += 1

    # inversion count over segment
    inversion_count = 0
    for r in range(len(mat)):
        row = mat[r]
        row_viol = 0
        for i in range(len(row) - 1):
            if np.isfinite(row[i]) and np.isfinite(row[i + 1]) and row[i] < row[i + 1]:
                row_viol += 1
        if row_viol > 0:
            inversion_count += 1

    return {
        "fan_spread_now": float(spread[-1]) if len(spread) else 0.0,
        "fan_spread_mean": float(np.nanmean(spread)) if len(spread) else 0.0,
        "fan_spread_slope": float(spread_reg["lin_slope"]),
        "fan_spread_accel": float(spread_reg["quad_curv"]),
        "fan_inversion_count": int(inversion_count),
        "fan_order_violation_now": int(viol_now),
    }


def find_last_g83_anchor(
    df: pd.DataFrame,
    current_idx: int,
    g83_col: str = "g_83",
    min_anchor_sep_points: int = 3,
) -> Optional[AnchorInfo]:
    """
    Find the most recent G83 peak or valley before current_idx.
    This is the segment anchor.
    """
    if g83_col not in df.columns:
        return None

    vals = pd.to_numeric(df[g83_col], errors="coerce").to_numpy()
    if current_idx < 2:
        return None

    for i in range(current_idx - 1, 0, -1):
        if current_idx - i < min_anchor_sep_points:
            continue

        if _is_local_peak(vals, i):
            return AnchorInfo(
                anchor_idx=i,
                anchor_epoch=int(df.iloc[i]["epoch"]),
                anchor_type="PEAK",
            )

        if _is_local_valley(vals, i):
            return AnchorInfo(
                anchor_idx=i,
                anchor_epoch=int(df.iloc[i]["epoch"]),
                anchor_type="VALLEY",
            )

    return None


def validate_g83_segment(
    df: pd.DataFrame,
    anchor_idx: int,
    current_idx: int,
    min_segment_points: int = 8,
    max_gap_ratio: float = 0.20,
    max_missing_bars_sum: int = 12,
) -> tuple[bool, dict]:
    """
    Validate that the segment from anchor -> current has enough density to use.
    Uses optional continuity columns if present.
    """
    if anchor_idx is None or current_idx is None or current_idx <= anchor_idx:
        return False, {"reason": "bad_indices"}

    seg = df.iloc[anchor_idx : current_idx + 1].copy()
    n = len(seg)
    if n < min_segment_points:
        return False, {"reason": "too_short", "segment_points": n}

    # Coverage diagnostics if available
    coverage_ok = True
    gap_ratio = 0.0
    missing_sum = 0

    if "coverage_ratio" in seg.columns:
        gap_ratio = float(1.0 - seg["coverage_ratio"].mean())
        if gap_ratio > max_gap_ratio:
            coverage_ok = False

    if "bars_missing" in seg.columns:
        missing_sum = int(seg["bars_missing"].sum())
        if missing_sum > max_missing_bars_sum:
            coverage_ok = False

    if not coverage_ok:
        return False, {
            "reason": "coverage_fail",
            "segment_points": n,
            "gap_ratio": gap_ratio,
            "missing_sum": missing_sum,
        }

    return True, {
        "reason": "ok",
        "segment_points": n,
        "gap_ratio": gap_ratio,
        "missing_sum": missing_sum,
    }


def determine_tail_start(
    df: pd.DataFrame,
    anchor_idx: int,
    current_idx: int,
    fast_cols: tuple[str, ...] = ("g_23", "g_8"),
    min_tail_points: int = 6,
    tail_search_frac: float = 0.33,
    min_extremum_sep_points: int = 2,
) -> TailInfo:
    """
    Adaptive tail definition.

    Primary:
      - search for the most recent internal local extremum on a faster Gaussian
        inside the last third of the anchor->current segment.

    Fallback:
      - use the last max(min_tail_points, round(segment_len * tail_search_frac)) points
    """
    seg_len = current_idx - anchor_idx + 1
    fallback_tail_len = max(min_tail_points, int(round(seg_len * tail_search_frac)))
    fallback_start = max(anchor_idx, current_idx - fallback_tail_len + 1)

    search_start = max(anchor_idx + 1, int(round(current_idx - seg_len * tail_search_frac)))

    for col in fast_cols:
        if col not in df.columns:
            continue

        vals = pd.to_numeric(df[col], errors="coerce").to_numpy()

        for i in range(current_idx - min_extremum_sep_points, search_start - 1, -1):
            if i <= anchor_idx + 1:
                continue

            if _is_local_peak(vals, i) or _is_local_valley(vals, i):
                if (current_idx - i + 1) >= min_tail_points:
                    return TailInfo(
                        tail_start_idx=i,
                        tail_method=f"adaptive_extremum:{col}",
                    )

    return TailInfo(
        tail_start_idx=fallback_start,
        tail_method="fallback_fraction",
    )


def extract_g83_segment_features(
    df: pd.DataFrame,
    anchor: AnchorInfo,
    tail: TailInfo,
    current_idx: int,
    sigma_cols: tuple[str, ...] = ("g_8", "g_23", "g_38", "g_53", "g_68", "g_83"),
) -> dict:
    """
    Extract segment-level and tail-level features from:
      anchor_idx -> current_idx
    and
      tail_start_idx -> current_idx
    """
    seg = df.iloc[anchor.anchor_idx : current_idx + 1].copy()
    tail_df = df.iloc[tail.tail_start_idx : current_idx + 1].copy()

    out = {
        "anchor_epoch": int(anchor.anchor_epoch),
        "anchor_type": anchor.anchor_type,
        "segment_len_epochs": int(len(seg)),
        "tail_len_epochs": int(len(tail_df)),
        "tail_method": tail.tail_method,
    }

    # Segment direction using G83 anchor->now
    if "g_83" in seg.columns:
        g83_seg = pd.to_numeric(seg["g_83"], errors="coerce").to_numpy(dtype=float)
        g83_tail = pd.to_numeric(tail_df["g_83"], errors="coerce").to_numpy(dtype=float)

        out["g83_anchor_to_now_delta"] = float(g83_seg[-1] - g83_seg[0]) if len(g83_seg) else 0.0
        out["segment_direction"] = (
            "UP" if out["g83_anchor_to_now_delta"] > 0 else "DOWN" if out["g83_anchor_to_now_delta"] < 0 else "FLAT"
        )

        seg_reg = _linreg_metrics(g83_seg)
        tail_reg = _linreg_metrics(g83_tail)
        seg_energy = _segment_energy(g83_seg)
        tail_energy = _segment_energy(g83_tail)

        out.update({
            "seg_g83_slope": seg_reg["lin_slope"],
            "seg_g83_r2": seg_reg["lin_r2"],
            "seg_g83_quad_tangent": seg_reg["quad_tangent"],
            "seg_g83_quad_curv": seg_reg["quad_curv"],

            "tail_g83_slope": tail_reg["lin_slope"],
            "tail_g83_r2": tail_reg["lin_r2"],
            "tail_g83_quad_tangent": tail_reg["quad_tangent"],
            "tail_g83_quad_curv": tail_reg["quad_curv"],

            "seg_energy_total": seg_energy["energy_total"],
            "seg_momentum_score": seg_energy["momentum_score"],
            "tail_energy_total": tail_energy["energy_total"],
            "tail_momentum_score": tail_energy["momentum_score"],

            "seg_plateau_score": _plateau_score(g83_seg),
            "tail_plateau_score": _plateau_score(g83_tail),
        })

    # Fast-vs-slow tail divergence
    if "g_23" in tail_df.columns and "g_83" in tail_df.columns:
        g23_tail = pd.to_numeric(tail_df["g_23"], errors="coerce").to_numpy(dtype=float)
        g83_tail = pd.to_numeric(tail_df["g_83"], errors="coerce").to_numpy(dtype=float)

        g23_reg = _linreg_metrics(g23_tail)
        g83_reg = _linreg_metrics(g83_tail)

        out["tail_fast_slow_slope_delta"] = float(g23_reg["lin_slope"] - g83_reg["lin_slope"])
        out["tail_fast_slow_curv_delta"] = float(g23_reg["quad_curv"] - g83_reg["quad_curv"])

    # Fan spread / inversion across segment
    fan_feats = _fan_spread_features(seg, list(sigma_cols))
    out.update(fan_feats)

    # Tail spread specifically
    tail_fan_feats = _fan_spread_features(tail_df, list(sigma_cols))
    out["tail_fan_spread_now"] = tail_fan_feats["fan_spread_now"]
    out["tail_fan_spread_slope"] = tail_fan_feats["fan_spread_slope"]
    out["tail_fan_spread_accel"] = tail_fan_feats["fan_spread_accel"]
    out["tail_fan_inversion_count"] = tail_fan_feats["fan_inversion_count"]
    out["tail_fan_order_violation_now"] = tail_fan_feats["fan_order_violation_now"]

    # Optional raw price context if available
    if "close" in seg.columns:
        close_seg = pd.to_numeric(seg["close"], errors="coerce").to_numpy(dtype=float)
        close_tail = pd.to_numeric(tail_df["close"], errors="coerce").to_numpy(dtype=float)

        out["price_anchor_to_now_delta"] = float(close_seg[-1] - close_seg[0]) if len(close_seg) else 0.0
        out["tail_price_delta"] = float(close_tail[-1] - close_tail[0]) if len(close_tail) else 0.0

    return out


def build_segment_feature_row(
    df: pd.DataFrame,
    current_idx: int,
    g83_col: str = "g_83",
) -> dict:
    """
    One-stop helper for current epoch:
      1. find last G83 anchor
      2. validate segment
      3. determine tail
      4. extract features
    """
    row = {
        "segment_valid": 0,
        "segment_reason": "uninitialized",
    }

    if current_idx <= 2:
        row["segment_reason"] = "too_early"
        return row

    anchor = find_last_g83_anchor(df, current_idx=current_idx, g83_col=g83_col)
    if anchor is None:
        row["segment_reason"] = "no_anchor"
        return row

    valid, meta = validate_g83_segment(df, anchor.anchor_idx, current_idx)
    if not valid:
        row["segment_reason"] = meta.get("reason", "segment_invalid")
        row.update(meta)
        row["anchor_epoch"] = anchor.anchor_epoch
        row["anchor_type"] = anchor.anchor_type
        return row

    tail = determine_tail_start(df, anchor.anchor_idx, current_idx)
    feats = extract_g83_segment_features(df, anchor, tail, current_idx)

    row["segment_valid"] = 1
    row["segment_reason"] = "ok"
    row.update(meta)
    row.update(feats)
    return row

def build_segment_feature_table(
    df: pd.DataFrame,
    g83_col: str = "g_83",
) -> pd.DataFrame:
    """
    Apply segment-engine logic row-by-row across the epoch dataframe.

    Returns a dataframe aligned to df.index containing:
      - segment_valid
      - anchor / tail metadata
      - segment / tail slope, regression, plateau, energy, spread, inversion, etc.

    This should be called AFTER gaussian_stack has been added, because it depends
    on columns like g_83, g_23, g_8, etc.
    """
    rows: list[dict] = []

    if df.empty:
        return pd.DataFrame(index=df.index)

    for current_idx in range(len(df)):
        feats = build_segment_feature_row(df, current_idx=current_idx, g83_col=g83_col)
        rows.append(feats)

    seg_df = pd.DataFrame(rows, index=df.index)

    # Normalize / encode some string fields for ML compatibility
    if "anchor_type" in seg_df.columns:
        seg_df["anchor_is_peak"] = (seg_df["anchor_type"] == "PEAK").astype(int)
        seg_df["anchor_is_valley"] = (seg_df["anchor_type"] == "VALLEY").astype(int)

    if "segment_direction" in seg_df.columns:
        seg_df["segment_dir_up"] = (seg_df["segment_direction"] == "UP").astype(int)
        seg_df["segment_dir_down"] = (seg_df["segment_direction"] == "DOWN").astype(int)
        seg_df["segment_dir_flat"] = (seg_df["segment_direction"] == "FLAT").astype(int)

    if "tail_method" in seg_df.columns:
        seg_df["tail_method_adaptive"] = seg_df["tail_method"].astype(str).str.startswith("adaptive_extremum").astype(int)
        seg_df["tail_method_fallback"] = (seg_df["tail_method"] == "fallback_fraction").astype(int)

    # keep strings too for debug / exports, but numeric columns are what the model will use
    return seg_df