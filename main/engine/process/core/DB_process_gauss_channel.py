# main/engine/process/DB_process_gauss_channel.py
"""
Gaussian Channel snapshot utilities.

This module is intentionally "process-friendly":
- It can build a *single-sigma* snapshot (old call pattern):
    build_gauss_channel_snapshot(sigma, channel, last_close, z_mid=None, z_width=None)

- Or it can build a *multi-sigma* snapshot payload (newer call pattern used by orchestrators):
    build_gauss_channel_snapshot(timing, windows, per_sigma_full, config, prev_snapshot=None)

Additionally, it can compute a single scalar:
    next_epoch_nb_likelihood  (Neutral/Bear likelihood for *next* epoch)

The scalar is designed for your "late Bull hookdown" situation:
micro turns down (sigma 8) while macro (sigma 23) is still up but fading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


# ----------------------------
# Data models
# ----------------------------

@dataclass
class ChannelSnapshot:
    """Single-sigma channel snapshot."""
    sigma: int
    mid_last: float
    upper_last: float
    lower_last: float
    width_last: float
    z_mid: Optional[float] = None
    z_width: Optional[float] = None


@dataclass
class ChannelStats:
    """Compact channel stats used by DB_process_printing.

    mid_last: last midline value (e.g., last Gaussian value)
    robust_std: robust dispersion estimate (MAD->sigma)
    lower/upper: mid_last ± k*robust_std
    width: upper-lower
    delta: mid_last - mid_first
    slope: delta / (n-1) for midline series
    tag: qualitative width trend label ('expanding'|'contracting'|'flat')
    """
    mid_last: float
    robust_std: float
    lower: float
    upper: float
    width: float
    delta: float
    slope: float
    tag: str = "flat"


# ----------------------------
# Robust statistics helpers (needed by DB_process_printing)
# ----------------------------

def robust_std(values: list) -> float:
    """Robust std estimate using MAD (median absolute deviation).

    Returns 0.0 if not enough finite numeric values.
    """
    import math
    if not values:
        return 0.0
    xs = []
    for v in values:
        try:
            fv = float(v)
            if fv == fv and math.isfinite(fv):
                xs.append(fv)
        except Exception:
            continue
    if len(xs) < 2:
        return 0.0
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    med = xs_sorted[n // 2] if n % 2 == 1 else 0.5 * (xs_sorted[n // 2 - 1] + xs_sorted[n // 2])
    devs = sorted(abs(x - med) for x in xs_sorted)
    mad = devs[n // 2] if n % 2 == 1 else 0.5 * (devs[n // 2 - 1] + devs[n // 2])
    # 1.4826 scales MAD to match sigma for normal distribution
    return 1.4826 * mad


def _extract_values(series_obj) -> list:
    """Best-effort extraction of numeric series values."""
    if series_obj is None:
        return []
    # common: {'values': [...]} or {'series': [...]} or {'mid': [...]}
    if isinstance(series_obj, dict):
        for k in ("values", "series", "mid", "gauss", "g"):
            if k in series_obj and isinstance(series_obj[k], list):
                return series_obj[k]
        # if dict itself maps idx->value
        if all(isinstance(v, (int, float)) for v in series_obj.values()):
            return list(series_obj.values())
        return []
    if isinstance(series_obj, list):
        return series_obj
    return []


def build_channel_snapshot(per_sigma_full: dict, k: float = 2.0) -> dict:
    """Build ChannelStats for each sigma from a full midline series.

    per_sigma_full: {sigma: series_obj}, where series_obj is either a list of
    values or a dict containing a list under 'values'/'series'/'mid'.

    Returns: {sigma: ChannelStats}
    """
    out = {}
    for sigma, series_obj in (per_sigma_full or {}).items():
        vals = _extract_values(series_obj)
        if not vals:
            continue
        mid_first = float(vals[0])
        mid_last = float(vals[-1])
        rs = robust_std(vals)
        upper = mid_last + k * rs
        lower = mid_last - k * rs
        width = upper - lower

        n = len(vals)
        delta = mid_last - mid_first
        slope = (delta / (n - 1)) if n > 1 else 0.0

        # width-trend tag: compare robust_std on first vs last half
        half = max(2, n // 2)
        rs_first = robust_std(vals[:half])
        rs_last = robust_std(vals[-half:])
        tag = "flat"
        if rs_first > 0 and rs_last > rs_first * 1.05:
            tag = "expanding"
        elif rs_first > 0 and rs_last < rs_first * 0.95:
            tag = "contracting"

        out[int(sigma)] = ChannelStats(
            mid_last=mid_last,
            robust_std=rs,
            lower=lower,
            upper=upper,
            width=width,
            delta=delta,
            slope=slope,
            tag=tag,
        )
    return out


def build_channel_pv_tail(*args, **kwargs):
    """Build PV-leg channel series (two legs) using causal rolling robust_std.

    NEW output keys:
      - per_sigma_leg1: prev → last (PV-leg)
      - per_sigma_leg2: last → now  (TAIL)
    Back-compat:
      - per_sigma remains and is set to per_sigma_leg2 (TAIL), so old printing still works.

    Existing schema reference is documented in this file.【turn10:11†DB_process_gauss_channel.py†L1-L21】
    """
    # Support keyword-style calls (preferred)
    timing = kwargs.get("timing", None)
    windows = kwargs.get("windows", None)
    per_sigma_hist = kwargs.get("per_sigma_hist", None)
    pv_ref_sigma = int(kwargs.get("pv_ref_sigma", 23))
    config = kwargs.get("config", {}) or {}

    # Support old positional signature: (timing, windows, per_sigma_hist, pv_ref_sigma, config)
    if per_sigma_hist is None and len(args) >= 3:
        timing = args[0]
        windows = args[1]
        per_sigma_hist = args[2]
    if len(args) >= 4:
        try:
            pv_ref_sigma = int(args[3])
        except Exception:
            pass
    if len(args) >= 5 and not config:
        config = args[4] or {}

    per_sigma_hist = per_sigma_hist or {}

    # Determine decision end time (best effort) — same approach as current code
    last_ts_candidates = []
    for _s, _pack in (per_sigma_hist or {}).items():
        ts_all = (_pack or {}).get("ts", []) or []
        if ts_all:
            try:
                last_ts_candidates.append(max(ts_all))
            except Exception:
                pass
    last_ts = max(last_ts_candidates) if last_ts_candidates else None
    if not last_ts:
        return {"status": "empty", "reason": "no_samples", "pv_ref_sigma": pv_ref_sigma, "per_sigma": {}}

    k = float(config.get("GAUSS_CHANNEL_K", 2.0))
    win_n = int(config.get("PV_TAIL_CHANNEL_WINDOW_N", 21))
    win_n = max(5, min(200, win_n))

    # Find pv_pair on pv_ref_sigma within bell lookback
    try:
        from main.engine.process.utils.DB_process_slicing import slice_by_window
        from DB_process_metrics import find_last_extrema_pair
    except Exception:
        from main.engine.process.utils.DB_process_slicing import slice_by_window
        from main.engine.process.core.DB_process_metrics import find_last_extrema_pair

    from datetime import timedelta
    lookback_minutes = int(config.get("BELL_CURVE_LOOKBACK_MINUTES", 240))
    start_dt = last_ts - timedelta(minutes=lookback_minutes)

    ref_pack = per_sigma_hist.get(pv_ref_sigma, {}) or {}
    ref_ts_all = ref_pack.get("ts", []) or []
    ref_vals_all = ref_pack.get("values", []) or []
    ref_ts_win, ref_vals_win = slice_by_window(ref_ts_all, ref_vals_all, start_dt, last_ts)

    pv_min_sep = float(config.get("PV_MIN_SEP_SECONDS", 10.0))
    pv_pair = find_last_extrema_pair(ref_ts_win, ref_vals_win, last_ts, min_sep_seconds=pv_min_sep)
    if pv_pair is None:
        return {
            "status": "failed",
            "reason": "pv_pair_not_found",
            "pv_ref_sigma": pv_ref_sigma,
            "pv_pair": None,
            "last_ts": last_ts,
            "k": k,
            "window_n": win_n,
            "per_sigma": {},
            "per_sigma_leg1": {},
            "per_sigma_leg2": {},
        }

    # Leg boundaries:
    #   prev -> last  (LEG1: PV-leg)
    #   last -> now   (LEG2: tail continuation)
    t_prev = pv_pair["prev"]["ts"]
    t_last = pv_pair["last"]["ts"]
    t_now = last_ts

    # Guard ordering (should already be ordered, but don’t assume)
    # Ensure: t_prev <= t_last <= t_now
    try:
        if t_last < t_prev:
            t_prev, t_last = t_last, t_prev
    except Exception:
        pass

    def _roll_stats(ts: list, vals: list) -> dict:
        mid = [float(v) for v in vals]

        rs_list = []
        for i in range(len(mid)):
            j0 = max(0, i - win_n + 1)
            rs_list.append(robust_std(mid[j0 : i + 1]))

        upper = [m + k * rs for m, rs in zip(mid, rs_list)]
        lower = [m - k * rs for m, rs in zip(mid, rs_list)]
        width = [u - l for u, l in zip(upper, lower)]

        d_mid = [0.0] + [mid[i] - mid[i - 1] for i in range(1, len(mid))]
        d_width = [0.0] + [width[i] - width[i - 1] for i in range(1, len(width))]

        return {
            "ts": ts,
            "mid": mid,
            "width": width,
            "upper": upper,
            "lower": lower,
            "d_mid": d_mid,
            "d_width": d_width,
        }

    out_leg1 = {}
    out_leg2 = {}

    # Build per-sigma legs
    for sigma, pack in (per_sigma_hist or {}).items():
        pack = pack or {}
        ts_all = pack.get("ts", []) or []
        vals_all = pack.get("values", []) or []
        if not ts_all or not vals_all or len(ts_all) != len(vals_all):
            continue

        # LEG1: t_prev → t_last
        ts1, v1 = slice_by_window(ts_all, vals_all, t_prev, t_last)
        if ts1 and v1 and len(ts1) >= 3:
            out_leg1[int(sigma)] = _roll_stats(ts1, v1)

        # LEG2: t_last → t_now
        ts2, v2 = slice_by_window(ts_all, vals_all, t_last, t_now)
        if ts2 and v2 and len(ts2) >= 3:
            out_leg2[int(sigma)] = _roll_stats(ts2, v2)

    # If nothing built, fail softly
    if not out_leg1 and not out_leg2:
        return {
            "status": "empty",
            "reason": "no_leg_samples",
            "pv_ref_sigma": pv_ref_sigma,
            "pv_pair": pv_pair,
            "last_ts": last_ts,
            "k": k,
            "window_n": win_n,
            "per_sigma": {},
            "per_sigma_leg1": {},
            "per_sigma_leg2": {},
        }

    # Backward compat: per_sigma := LEG2 (tail continuation)
    return {
        "status": "ok",
        "pv_ref_sigma": pv_ref_sigma,
        "pv_pair": pv_pair,
        "leg1": {"start": t_prev, "end": t_last},
        "leg2": {"start": t_last, "end": t_now},
        "last_ts": last_ts,
        "k": k,
        "window_n": win_n,
        "per_sigma": out_leg2,          # legacy behavior expects this key
        "per_sigma_leg1": out_leg1,     # new
        "per_sigma_leg2": out_leg2,     # new (same as per_sigma)
    }

# ----------------------------
# Existing snapshot builders + NB likelihood scalar
# (kept intact below)
# ----------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_gauss_channel_snapshot(*args, **kwargs):
    """
    Dual-call function:

    (A) old style:
        build_gauss_channel_snapshot(sigma, channel, last_close, z_mid=None, z_width=None)

    (B) new style:
        build_gauss_channel_snapshot(timing, windows, per_sigma_full, config, prev_snapshot=None)

    This file was originally written to support both, because different parts of your
    process pipeline evolved at different times.
    """
    # --- detect old-style signature: (sigma, channel, last_close, ...)
    if len(args) >= 3 and isinstance(args[0], (int, float)):
        sigma = int(args[0])
        channel = args[1]
        last_close = _safe_float(args[2])
        z_mid = kwargs.get("z_mid", None)
        z_width = kwargs.get("z_width", None)

        # channel may already provide upper/lower/mid
        mid_last = _safe_float(getattr(channel, "mid_last", None), default=last_close)
        upper_last = _safe_float(getattr(channel, "upper_last", None), default=mid_last)
        lower_last = _safe_float(getattr(channel, "lower_last", None), default=mid_last)
        width_last = _safe_float(getattr(channel, "width_last", None), default=(upper_last - lower_last))

        return ChannelSnapshot(
            sigma=sigma,
            mid_last=mid_last,
            upper_last=upper_last,
            lower_last=lower_last,
            width_last=width_last,
            z_mid=z_mid,
            z_width=z_width,
        )

    # --- new style signature
    timing = args[0] if len(args) > 0 else kwargs.get("timing")
    windows = args[1] if len(args) > 1 else kwargs.get("windows")
    per_sigma_full = args[2] if len(args) > 2 else kwargs.get("per_sigma_full")
    config = args[3] if len(args) > 3 else kwargs.get("config")
    prev_snapshot = kwargs.get("prev_snapshot", None)

    # expected config keys (safe defaults)
    k = _safe_float((config or {}).get("k", 2.0), 2.0)
    sigma_micro = int((config or {}).get("sigma_micro", 8))
    sigma_macro = int((config or {}).get("sigma_macro", 23))

    # Build stats for printing and for scalar
    stats = build_channel_snapshot(per_sigma_full or {}, k=k)

    # Compute the NB likelihood scalar if possible
    nb = compute_next_epoch_nb_likelihood(
        per_sigma_full=per_sigma_full or {},
        stats=stats,
        sigma_micro=sigma_micro,
        sigma_macro=sigma_macro,
        prev_snapshot=prev_snapshot,
        config=config or {},
    )

    payload = {
        "timing": timing,
        "windows": windows,
        "k": k,
        "sigma_micro": sigma_micro,
        "sigma_macro": sigma_macro,
        "snapshots": stats,  # ChannelStats per sigma (printing-friendly)
        "next_epoch_nb_likelihood": nb,
    }
    return payload


def compute_next_epoch_nb_likelihood(
    per_sigma_full: Dict[int, Any],
    stats: Dict[int, ChannelStats],
    sigma_micro: int = 8,
    sigma_macro: int = 23,
    prev_snapshot: Optional[dict] = None,
    config: Optional[dict] = None,
) -> float:
    """
    Next epoch Neutral/Bear likelihood scalar (S).

    S = α·max(0, -V8)      micro down
      + β·max(0,  V23)     macro still up
      + γ·max(0, -a23)     macro fading (deceleration)
      + δ·max(0, -Δw8)     micro contraction
      + ε·p8(t)            micro exhaustion (simple proxy)

    Returned scalar is clamped to [0, 1] to behave like a "likelihood-ish" score.
    """
    cfg = config or {}

    alpha = _safe_float(cfg.get("nb_alpha", 1.0), 1.0)
    beta  = _safe_float(cfg.get("nb_beta", 0.35), 0.35)
    gamma = _safe_float(cfg.get("nb_gamma", 0.65), 0.65)
    delta = _safe_float(cfg.get("nb_delta", 0.45), 0.45)
    eps   = _safe_float(cfg.get("nb_eps", 0.30), 0.30)

    # Extract midline values for micro/macro
    v8_series = _extract_values(per_sigma_full.get(sigma_micro))
    v23_series = _extract_values(per_sigma_full.get(sigma_macro))

    if len(v8_series) < 3 or len(v23_series) < 3:
        return 0.0

    # Velocities (simple slope per sample)
    V8 = _safe_float(v8_series[-1]) - _safe_float(v8_series[-2])
    V23 = _safe_float(v23_series[-1]) - _safe_float(v23_series[-2])

    # Macro acceleration proxy: last delta - previous delta
    d23_now = _safe_float(v23_series[-1]) - _safe_float(v23_series[-2])
    d23_prev = _safe_float(v23_series[-2]) - _safe_float(v23_series[-3])
    a23 = d23_now - d23_prev

    # Δw8: compare micro width to previous snapshot width if available
    w8_now = stats.get(sigma_micro).width if sigma_micro in stats else 0.0
    w8_prev = 0.0
    if prev_snapshot and isinstance(prev_snapshot, dict):
        try:
            prev_stats = prev_snapshot.get("snapshots", {})
            if isinstance(prev_stats, dict) and sigma_micro in prev_stats:
                # may be dict or ChannelStats
                p = prev_stats[sigma_micro]
                w8_prev = _safe_float(getattr(p, "width", None), default=_safe_float(p.get("width", 0.0) if isinstance(p, dict) else 0.0))
        except Exception:
            w8_prev = 0.0
    dW8 = w8_now - w8_prev

    # p8(t) exhaustion proxy: large recent move relative to robust dispersion
    rs8 = stats.get(sigma_micro).robust_std if sigma_micro in stats else 0.0
    move8 = abs(_safe_float(v8_series[-1]) - _safe_float(v8_series[-3]))
    p8 = 0.0
    if rs8 > 0:
        p8 = _clamp(move8 / (3.0 * rs8), 0.0, 1.0)

    S = (
        alpha * max(0.0, -V8) +
        beta  * max(0.0,  V23) +
        gamma * max(0.0, -a23) +
        delta * max(0.0, -dW8) +
        eps   * p8
    )

    # normalize-ish: squash with a simple clamp after scaling
    # (keeps it interpretable without requiring a training pass)
    scale = _safe_float(cfg.get("nb_scale", 1.0), 1.0)
    return _clamp(S * scale, 0.0, 1.0)
