from __future__ import annotations

from typing import Any, Dict, List


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _tail(values: List[float], n: int) -> List[float]:
    if not values:
        return []
    return values[-n:] if len(values) > n else list(values)


def _slope(values: List[float], k: int) -> float:
    if len(values) < 2:
        return 0.0
    kk = max(1, min(k, len(values) - 1))
    return (_safe_float(values[-1]) - _safe_float(values[-1 - kk])) / float(kk)


def _curvature(values: List[float], k_short: int = 5, k_long: int = 21) -> float:
    return _slope(values, k_short) - _slope(values, k_long)


def build_feature_catalog(
    *,
    timing: Any,
    per_sigma_hist: Dict[int, Dict[str, Any]],
    config: Dict[str, Any] | None = None,
    hyst_obj: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Single-source feature dictionary consumed by all trend models."""
    config = config or {}
    hyst_obj = hyst_obj or {}

    sigma_values: Dict[int, List[float]] = {}
    for sigma, pack in (per_sigma_hist or {}).items():
        vals = (pack or {}).get("values") or []
        sigma_values[int(sigma)] = [float(x) for x in vals if x is not None]

    def last(s: int) -> float:
        vals = sigma_values.get(s, [])
        return _safe_float(vals[-1]) if vals else 0.0

    s8 = sigma_values.get(8, [])
    s23 = sigma_values.get(23, [])
    s38 = sigma_values.get(38, [])
    s53 = sigma_values.get(53, [])
    s68 = sigma_values.get(68, [])
    s83 = sigma_values.get(83, [])

    fan_width = last(83) - last(8)
    spacing_8_23 = last(23) - last(8)
    spacing_23_53 = last(53) - last(23)
    spacing_53_83 = last(83) - last(53)

    denom = abs(spacing_23_53) + 1e-9
    ratio_8_23_to_23_53 = spacing_8_23 / denom
    ratio_53_83_to_23_53 = spacing_53_83 / denom

    slope_8 = _slope(s8, 8)
    slope_23 = _slope(s23, 8)
    slope_38 = _slope(s38, 8)
    slope_53 = _slope(s53, 8)
    slope_68 = _slope(s68, 8)
    slope_83 = _slope(s83, 8)

    tan_8 = _slope(s8, 3)
    tan_23 = _slope(s23, 3)
    tan_38 = _slope(s38, 3)
    tan_53 = _slope(s53, 3)
    tan_68 = _slope(s68, 3)
    tan_83 = _slope(s83, 3)

    curv_8 = _curvature(s8)
    curv_23 = _curvature(s23)
    curv_38 = _curvature(s38)
    curv_53 = _curvature(s53)
    curv_68 = _curvature(s68)
    curv_83 = _curvature(s83)

    compression_now = abs(spacing_8_23) + abs(spacing_23_53) + abs(spacing_53_83)
    h8 = _tail(s8, 20)
    h23 = _tail(s23, 20)
    h53 = _tail(s53, 20)
    h83 = _tail(s83, 20)
    compression_prev = 0.0
    if h8 and h23 and h53 and h83:
        compression_prev = abs(h23[0] - h8[0]) + abs(h53[0] - h23[0]) + abs(h83[0] - h53[0])

    compression_velocity = compression_now - compression_prev

    slope_signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in [slope_8, slope_23, slope_53, slope_83]]
    aligned = len(set(slope_signs)) == 1 and slope_signs[0] != 0
    torque_alignment = 1.0 if aligned else 0.0
    flip_score = abs(curv_23) + abs(curv_53)

    # Optional contextual anchors (safe fallback to None when unavailable).
    pv_pair = None
    pv_ref_sigma = 23
    pv_blob = per_sigma_hist.get(pv_ref_sigma, {}) or {}
    pv_ts = (pv_blob.get("ts") or [])
    pv_vals = (pv_blob.get("values") or [])
    if pv_ts and pv_vals:
        try:
            from DB_process_metrics import find_last_extrema_pair
        except Exception:
            from main.engine.process.DB_process_metrics import find_last_extrema_pair

        try:
            pv_pair = find_last_extrema_pair(
                pv_ts,
                pv_vals,
                getattr(timing, "full_end", None) or getattr(timing, "dt_curr", None),
                min_sep_seconds=float((config or {}).get("PV_MIN_SEPARATION_SECONDS", 10.0)),
            )
        except Exception:
            pv_pair = None

    return {
        "meta": {
            "curr_epoch": int(getattr(timing, "curr_epoch", 0) or 0),
            "next_epoch": int(getattr(timing, "next_epoch", 0) or 0),
            "timestamp": str(getattr(timing, "dt_curr", "")),
            "decision_time": str(getattr(timing, "dt_curr", "")),
            "next_epoch_time": str(getattr(timing, "next_epoch_time", "")),
            "full_window_start": str(getattr(timing, "full_start", "")),
            "full_window_end": str(getattr(timing, "full_end", "")),
        },
        "gauss": {
            "latest": {
                "s8": last(8), "s23": last(23), "s38": last(38),
                "s53": last(53), "s68": last(68), "s83": last(83)
            },
            "slopes": {
                "s8": slope_8, "s23": slope_23, "s38": slope_38,
                "s53": slope_53, "s68": slope_68, "s83": slope_83
            },
            "tangent": {
                "s8": tan_8, "s23": tan_23, "s38": tan_38,
                "s53": tan_53, "s68": tan_68, "s83": tan_83
            },
            "curvature": {
                "s8": curv_8, "s23": curv_23, "s38": curv_38,
                "s53": curv_53, "s68": curv_68, "s83": curv_83
            },
        },
        "msbc": {
            "mid_bias": _safe_float((slope_23 + slope_53) / 2.0),
            "edge_bias": _safe_float((slope_8 + slope_83) / 2.0),
        },
        "gcs": {
            "slope_spread": max(slope_8, slope_23, slope_53, slope_83) - min(slope_8, slope_23, slope_53, slope_83),
            "fan_direction": 1 if fan_width > 0 else (-1 if fan_width < 0 else 0),
        },
        "spacing": {
            "spacing_8_23": spacing_8_23,
            "spacing_23_53": spacing_23_53,
            "spacing_53_83": spacing_53_83,
            "ratio_8_23_to_23_53": ratio_8_23_to_23_53,
            "ratio_53_83_to_23_53": ratio_53_83_to_23_53,
        },
        "fan": {
            "width": fan_width,
            "width_abs": abs(fan_width),
            "width_slope": _slope([last(83) - last(8) for _ in range(2)], 1),
        },
        "hysteresis": {
            "vote_raw": _safe_float((hyst_obj or {}).get("vote_raw", 0.0)),
            "flip_score": flip_score,
        },
        "torque": {
            "alignment": torque_alignment,
            "slope_signs": slope_signs,
        },
        "compression": {
            "value": compression_now,
            "velocity": compression_velocity,
        },
        "context": {
            "tail_anchor_type": (pv_pair or {}).get("last", {}).get("kind"),
            "extrema_pair": (pv_pair or {}).get("swing"),
            "pv_direction": (pv_pair or {}).get("swing"),
        },
    }