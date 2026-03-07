# main/engine/process/printing/DB_process_printing_hyst.py

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .DB_process_printing_utils import (
    _safe_get,
    _fmt_time,
    _line,
)
from main.engine.process.printing.DB_process_SectionLogger import get_section_logger

logger = logging.getLogger(__name__)

# ======================================================================================
# PHASE 1: MODULE-LEVEL TRACKER
# - Printing module owns this for now (simple + low-risk)
# - Phase 2: pass tracker from orchestrator for cleaner state management
# ======================================================================================

_HYST_TRACKER: Dict[str, Any] = {
    "episode_start_ts": None,
    "episode_pair": None,
    "last_primary_flip_ts": None,
    "leader_state": {},
}


# ======================================================================================
# PUBLIC EXPORTS (for DB_process_calc / DB_process_trend later)
# ======================================================================================

def get_hyst_tracker() -> Dict[str, Any]:
    """Phase 1: allow other modules to read tracker if needed."""
    return _HYST_TRACKER


# ======================================================================================
# SMALL NUMERIC HELPERS
# ======================================================================================

def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _mad(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)) + 1e-12)


def _percentile(arr: np.ndarray, p: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, p))


def _compute_slope(ts: List[datetime], vals: np.ndarray) -> float:
    """Simple linear slope vs seconds. vals may contain NaN (caller should pre-filter)."""
    if len(ts) < 5:
        return 0.0
    x = np.array([(t - ts[0]).total_seconds() for t in ts], dtype=float)
    y = vals.astype(float)
    if not np.isfinite(y).any():
        return 0.0
    # Filter NaNs
    m = np.isfinite(y)
    if m.sum() < 5:
        return 0.0
    x2 = x[m]
    y2 = y[m]
    try:
        return float(np.polyfit(x2, y2, 1)[0])
    except Exception:
        return 0.0


def _compute_accel(ts: List[datetime], vals: np.ndarray) -> float:
    """
    Quadratic acceleration vs seconds (2nd derivative).
    Uses y = a*x^2 + b*x + c => y'' = 2a.
    vals may contain NaN (caller may pass unfiltered).
    """
    if len(ts) < 7:
        return 0.0
    x = np.array([(t - ts[0]).total_seconds() for t in ts], dtype=float)
    y = vals.astype(float)
    if not np.isfinite(y).any():
        return 0.0
    m = np.isfinite(y)
    if m.sum() < 7:
        return 0.0
    x2 = x[m]
    y2 = y[m]
    try:
        a = float(np.polyfit(x2, y2, 2)[0])
        return float(2.0 * a)
    except Exception:
        return 0.0


def _find_last_sign_flip(ts: List[datetime], vals: np.ndarray) -> Optional[datetime]:
    """Find most recent index where sign changes between two adjacent non-zero signs."""
    if len(ts) < 3 or vals.size < 3:
        return None
    signs = np.array([_sign(v) for v in vals], dtype=int)
    for i in range(len(signs) - 1, 0, -1):
        a = signs[i - 1]
        b = signs[i]
        if a != 0 and b != 0 and a != b:
            return ts[i]
    return None


def _nearest_align(
        base_ts: List[datetime],
        other_ts: List[datetime],
        other_vals: np.ndarray,
        tol_seconds: float = 3.0,
) -> np.ndarray:
    """
    Nearest neighbor alignment (Phase 1).
    Returns array aligned to base_ts; NaN if no match within tolerance.
    """
    out = np.full(len(base_ts), np.nan, dtype=float)
    if len(base_ts) == 0 or len(other_ts) == 0:
        return out

    j = 0
    for i, t in enumerate(base_ts):
        best_k = None
        best_dt = None

        # Move j forward while other_ts[j] is behind t (small optimization)
        while j + 1 < len(other_ts) and other_ts[j + 1] <= t:
            j += 1

        # Check around j (j and j+1) for nearest
        for k in (j, j + 1):
            if 0 <= k < len(other_ts):
                dt = abs((other_ts[k] - t).total_seconds())
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_k = k

        if best_k is not None and best_dt is not None and best_dt <= tol_seconds:
            out[i] = float(other_vals[best_k])

    return out


# ======================================================================================
# CORE FEATURE BUILDER (NO TREND DECISION)
# ======================================================================================

def compute_hysteresis_features(
        per_sigma_hist: Dict[int, Dict[str, Any]],
        decision_dt: datetime,
        lookback_seconds: int = 3600,
        tail_seconds: int = 120,
        align_tol_seconds: float = 3.0,
        primary_default: Tuple[int, int] = (38, 83),
        primary_slow: Tuple[int, int] = (53, 83),
        probe_pair: Tuple[int, int] = (23, 83),
) -> Dict[str, Any]:
    """
    Compute hysteresis episode + ladder stack metrics.
    DOES NOT produce Bull/Bear/Neutral or confidence.
    Returns a hyst_obj dict usable by trend/calc modules.
    """

    # ---- basic guards
    required = {23, 38, 53, 68, 83}
    missing = sorted([s for s in required if s not in per_sigma_hist])
    if missing:
        return {
            "meta": {
                "skipped": True,
                "skip_reason": f"missing_sigmas:{missing}",
                "missing_sigmas": missing,
            }
        }

    def _series(sigma: int) -> Tuple[List[datetime], np.ndarray]:
        ts = per_sigma_hist[sigma]["ts"]
        vals = np.array(per_sigma_hist[sigma]["values"], dtype=float)
        return ts, vals

    # base grid = g83 timestamps (slowest)
    ts83, g83 = _series(83)

    # baseline window mask
    cutoff = decision_dt - timedelta(seconds=lookback_seconds)
    mask_base = np.array([t >= cutoff for t in ts83], dtype=bool)

    # ---- choose primary pair (38,83) unless no crossing seen in lookback -> (53,83)
    mode = "DEFAULT"
    pair = primary_default

    def _d_pair(a: int, b: int) -> Tuple[np.ndarray, Optional[datetime]]:
        ts_a, g_a = _series(a)
        g_a_al = _nearest_align(ts83, ts_a, g_a, tol_seconds=align_tol_seconds)
        d = g_a_al - g83  # b assumed 83 base here
        last_flip = _find_last_sign_flip(ts83, d[mask_base])
        # last_flip found in masked array uses masked indices; simplest: recompute on full array:
        last_flip_full = _find_last_sign_flip(ts83, d)
        return d, last_flip_full

    d38_83, last_flip_38 = _d_pair(38, 83)

    # "slow switch": if there is no recent crossing (within lookback), we use (53,83)
    # We approximate that by: no crossing found at all OR last_flip is older than cutoff.
    if last_flip_38 is None or last_flip_38 < cutoff:
        d53_83, last_flip_53 = _d_pair(53, 83)
        if last_flip_53 is not None:
            pair = primary_slow
            d_primary = d53_83
            last_cross = last_flip_53
            mode = "SLOW_SWITCH"
        else:
            d_primary = d38_83
            last_cross = last_flip_38
    else:
        d_primary = d38_83
        last_cross = last_flip_38

    # Episode start anchor = last crossing within lookback if available
    start_ts = last_cross if (last_cross is not None and last_cross >= cutoff) else None
    elapsed = (decision_dt - start_ts).total_seconds() if start_ts else None
    stage = "IN_PROGRESS"  # true stage classification belongs in DB_process_trend/calc

    # ---- probe pair (23,83) flags (no trend decision, only signals)
    ts23, g23 = _series(probe_pair[0])
    g23_al = _nearest_align(ts83, ts23, g23, tol_seconds=align_tol_seconds)
    d23_83 = g23_al - g83
    probe_last_flip = _find_last_sign_flip(ts83, d23_83)

    flip_watch = False
    if start_ts and probe_last_flip and probe_last_flip > start_ts:
        # probe flipped after episode start (primary not necessarily ended yet)
        flip_watch = True

    # fast collapse flag (simple Phase 1 proxy): |d_probe| slope strongly negative over tail
    tail_cut = decision_dt - timedelta(seconds=tail_seconds)
    mask_tail = np.array([t >= tail_cut for t in ts83], dtype=bool)
    ts_tail = [ts83[i] for i in range(len(ts83)) if mask_tail[i]]
    dprobe_tail = np.abs(d23_83[mask_tail])
    dprobe_slope = _compute_slope(ts_tail, dprobe_tail)  # negative means collapsing
    fast_collapse = bool(dprobe_slope < 0)

    # ---- ladder stacks metrics (raw features; no vote/fusion)
    ladder_defs = {
        "S0": [23, 38, 53, 68, 83],
        "S1": [38, 53, 68, 83],
        "S2": [53, 68, 83],
        "S3": [68, 83],
    }

    stacks: Dict[str, Any] = {}
    for sid, sigs in ladder_defs.items():
        series_map = {}
        for s in sigs:
            ts_s, g_s = _series(s)
            series_map[s] = _nearest_align(ts83, ts_s, g_s, tol_seconds=align_tol_seconds)

        vals_stack = np.vstack([series_map[s] for s in sigs])  # shape: (k, n)
        mean_curve = np.nanmean(vals_stack, axis=0)

        # tail series
        mean_tail = mean_curve[mask_tail]
        m = _compute_slope(ts_tail, mean_tail)
        m2 = _compute_accel(ts_tail, mean_tail)

        # baseline drift scale (MAD of mean curve changes over baseline)
        mean_base = mean_curve[mask_base]
        m_scale = _mad(mean_base)
        m_norm = float(m / (m_scale + 1e-12)) if np.isfinite(m_scale) else 0.0
        m2_norm = float(m2 / (m_scale + 1e-12)) if np.isfinite(m_scale) else 0.0

        # spread series
        spread_series = np.nanmax(vals_stack, axis=0) - np.nanmin(vals_stack, axis=0)
        W_now = float(spread_series[-1]) if np.isfinite(spread_series[-1]) else float("nan")

        spread_base = spread_series[mask_base]
        W_p25 = _percentile(spread_base, 25)
        W_p75 = _percentile(spread_base, 75)

        W_med = float(np.median(spread_base)) if spread_base.size else 0.0
        W_mad = _mad(spread_base)
        W_z = float((W_now - W_med) / (W_mad + 1e-12)) if np.isfinite(W_now) else float("nan")
        W_ratio = float(W_now / (W_p75 + 1e-12)) if np.isfinite(W_now) else float("nan")

        # percentile position (0..100) of current W vs baseline
        if spread_base.size > 10 and np.isfinite(W_now):
            W_pct = float(np.mean(spread_base <= W_now) * 100.0)
        else:
            W_pct = float("nan")

        # spread velocity on tail
        spread_tail = spread_series[mask_tail]
        dW = _compute_slope(ts_tail, spread_tail)
        ddW = _compute_accel(ts_tail, spread_tail)

        dW_scale = _mad(spread_base)
        dW_norm = float(dW / (dW_scale + 1e-12)) if np.isfinite(dW_scale) else 0.0
        ddW_norm = float(ddW / (dW_scale + 1e-12)) if np.isfinite(dW_scale) else 0.0

        # order scores (adjacent inequalities at last point)
        last_vals = vals_stack[:, -1]
        # need all finite to assess order reliably
        if np.all(np.isfinite(last_vals)):
            bull_ok = np.all(np.diff(last_vals) < 0)  # g_fastest > g_next > ...
            bear_ok = np.all(np.diff(last_vals) > 0)
            order_bull = 1.0 if bull_ok else 0.0
            order_bear = 1.0 if bear_ok else 0.0

            # dominance / leaders (captures "G83 on top vs bottom" style polarity)
            idx_max = int(np.argmax(last_vals))
            idx_min = int(np.argmin(last_vals))
            leader_max_sigma = int(sigs[idx_max])
            leader_min_sigma = int(sigs[idx_min])
            dom = float((np.nanmax(last_vals) - np.nanmin(last_vals)) / (m_scale + 1e-12))
        else:
            order_bull = 0.0
            order_bear = 0.0
            leader_max_sigma = None
            leader_min_sigma = None
            dom = 0.0

        # simple cross-rate: count sign flips in adjacent differences over tail
        cross_count = 0
        pairs = list(zip(sigs[:-1], sigs[1:]))
        cross_pairs: Dict[str, Any] = {}
        for a, b in pairs:
            da = series_map[a][mask_tail] - series_map[b][mask_tail]
            sgn = np.array([_sign(x) for x in da], dtype=int)
            # count non-zero sign changes
            c = 0
            last_flip_ts = None
            for i in range(1, len(sgn)):
                if sgn[i] != 0 and sgn[i - 1] != 0 and sgn[i] != sgn[i - 1]:
                    cross_count += 1
                    c += 1
                    if i < len(ts_tail):
                        last_flip_ts = ts_tail[i]
            cross_pairs[f"{a}-{b}"] = {"count": int(c), "last_flip_ts": last_flip_ts}
        tail_minutes = max(tail_seconds / 60.0, 1e-9)
        cross_rate = float(cross_count / tail_minutes)

        # stability proxy: 1/(1+cross_rate)
        order_stability = float(1.0 / (1.0 + cross_rate))
        eff_order = float(max(order_bull, order_bear) * order_stability)

        # "ok" threshold for eff_order relative to baseline (p60 of eff_order baseline)
        # Phase 1 baseline: approximate using spread tightness as proxy when computing eff_order history is expensive
        # -> return eff_order + let trend/calc decide thresholds using full baseline later
        eff_order_ok = None  # leave decision to DB_process_trend/calc

        stacks[sid] = {
            "sigmas": sigs,
            "m": float(m),
            "m_norm": float(m_norm),
            "m2": float(m2),
            "m2_norm": float(m2_norm),
            "W_now": W_now,
            "W_pct": W_pct,
            "W_p25": W_p25,
            "W_p75": W_p75,
            "W_med": float(W_med),
            "W_mad": float(W_mad),
            "W_z": float(W_z),
            "W_ratio": float(W_ratio),
            "dW": float(dW),
            "dW_norm": float(dW_norm),
            "ddW": float(ddW),
            "ddW_norm": float(ddW_norm),
            "order_bull": float(order_bull),
            "order_bear": float(order_bear),
            "order_stability": float(order_stability),
            "eff_order": float(eff_order),
            "eff_order_ok": eff_order_ok,
            "cross_rate": float(cross_rate),
            "cross_pairs": cross_pairs,
            "dom": float(dom),
            "leader_max_sigma": leader_max_sigma,
            "leader_min_sigma": leader_min_sigma,
            "valid": True,
            "notes": "",
        }

    # ---- optional ETA metric (purely descriptive, not a decision)
    # Use S1 by default as a stable source for collapse ETA if dW < 0 and W meaningful
    eta_to_end_seconds = None
    eta_source = None
    s1 = stacks.get("S1")
    if s1 and np.isfinite(s1["W_now"]) and s1["dW"] < 0:
        eta_to_end_seconds = float(abs(s1["W_now"] / max(abs(s1["dW"]), 1e-12)))
        eta_source = "S1"

    hyst_obj = {
        "method_id": "method_hyster_v1.0",
        "episode": {
            "pair_used": [pair[0], pair[1]],
            "mode": mode,
            "lookback_seconds": lookback_seconds,
            "start_ts": start_ts,
            "last_cross_ts": last_cross,
            "elapsed_seconds": elapsed,
            "stage": stage,
            "d_now": float(d_primary[-1]) if d_primary.size else float("nan"),
            "sign_now": int(_sign(float(d_primary[-1]))) if d_primary.size else 0,
        },
        "probe": {
            "pair": [probe_pair[0], probe_pair[1]],
            "d_now": float(d23_83[-1]) if d23_83.size else float("nan"),
            "sign_now": int(_sign(float(d23_83[-1]))) if d23_83.size else 0,
            "flip_watch": bool(flip_watch),
            "fast_collapse": bool(fast_collapse),
            "d_abs_slope_tail": float(dprobe_slope),
            "last_flip_ts": probe_last_flip,
        },
        "eta": {
            "eta_to_end_seconds": eta_to_end_seconds,
            "source_stack": eta_source,
        },
        "stacks": stacks,
        "meta": {
            "decision_time": decision_dt,
            "tail_window_seconds": tail_seconds,
            "baseline_window_seconds": lookback_seconds,
            "align_tol_seconds": align_tol_seconds,
            "missing_sigmas": [],
            "skipped": False,
            "skip_reason": None,
        }
    }

    return hyst_obj


# ======================================================================================
# PRINTING ENTRYPOINT (LOG-ONLY)
# ======================================================================================

def print_hysteresis_fan_stack(
        timing: Any = None,
        windows: Any = None,
        decision_dt: Optional[datetime] = None,
        catalog: Any = None,
        config: Any = None,
        **_kwargs,
) -> Dict[str, Any]:
    """
    Logging-only printer.
    Returns hyst_obj features for downstream trend/calc code (but does not decide trend).
    """

    if not config or not config.get("LOG_HYSTERESIS", True):
        return {}

    slog = get_section_logger(logger, config)

    if decision_dt is None:
        decision_dt = datetime.now()

    per_sigma_hist = _safe_get(catalog, "per_sigma_hist", default={})
    if not per_sigma_hist:
        return {"meta": {"skipped": True, "skip_reason": "no_sigma_data"}}

    hyst = compute_hysteresis_features(
        per_sigma_hist=per_sigma_hist,
        decision_dt=decision_dt,
        lookback_seconds=3600,
        tail_seconds=120,
        align_tol_seconds=3.0,
    )

    if hyst.get("meta", {}).get("skipped"):
        return hyst

    # ----------------------------------------------------------------------------------
    # LOG BLOCK (matches the layout we designed)
    # ----------------------------------------------------------------------------------
    ep = hyst["episode"]
    pr = hyst["probe"]
    eta = hyst["eta"]
    stacks = hyst["stacks"]
    p = (config.get("PRINT", {}) or {})
    hcfg = (p.get("HYSTERESIS", {}) or {})
    hcfg_alt = (p.get("HYST", {}) or {})

    def _h_on(key: str, legacy: str, default: bool = True) -> bool:
        node = hcfg.get(key, hcfg_alt.get(key, {})) or {}
        return bool(node.get("ENABLED", config.get(legacy, default)))

    sec_stability = _h_on("STABILITY", "LOG_HYST_STABILITY", True)
    sec_pressure = _h_on("PRESSURE", "LOG_HYST_PRESSURE", True)
    sec_spread = _h_on("SPREAD_STATE", "LOG_HYST_SPREAD_STATE", True)
    sec_risk = _h_on("RISK", "LOG_HYST_RISK", True)

    slog.HYST_HEADER(_line("=", 155))
    slog.HYST_HEADER("HYSTERESIS FAN STACK (method_hyster_v1.0)")
    slog.HYST_HEADER(_line("-", 155))

    if True:
        slog.HYST_EPISODE(
            f"[hyst_episode] decision_time={_fmt_time(decision_dt)} | "
            f"pair=({ep['pair_used'][0]},{ep['pair_used'][1]}) | "
            f"start={_fmt_time(ep['start_ts']) if ep['start_ts'] else 'NONE'} | "
            f"elapsed={(ep['elapsed_seconds'] if ep['elapsed_seconds'] is not None else 0.0):.1f}s | "
            f"stage={ep['stage']}"
        )
        slog.HYST_EPISODE(
            f"[hyst_primary] d{ep['pair_used'][0]}_{ep['pair_used'][1]} now={ep['d_now']:+.6f} | "
            f"sign={ep['sign_now']} | "
            f"last_cross={_fmt_time(ep['last_cross_ts']) if ep['last_cross_ts'] else 'NONE'} | "
            f"lookback=60m | mode={ep['mode']}"
        )

    if True:
        slog.HYST_PROBE(
            f"[hyst_probe] pair=(23,83) | d23_83 now={pr['d_now']:+.6f} | "
            f"sign={pr['sign_now']} | flip_watch={int(pr['flip_watch'])} | "
            f"fast_collapse={int(pr['fast_collapse'])} | d_abs_slope_tail={pr['d_abs_slope_tail']:+.6f}"
        )

    if eta.get("eta_to_end_seconds") is not None:
        slog.HYST_ETA(
            f"[hyst_eta] eta={eta['eta_to_end_seconds']:.1f}s (~{eta['eta_to_end_seconds'] / 300.0:.2f} epochs) | "
            f"source={eta.get('source_stack')}"
        )

    if True:
        for sid in ["S0", "S1", "S2", "S3"]:
            lm = stacks.get(sid)
            if not lm:
                continue
            sigs = ",".join(map(str, lm["sigmas"]))
            W_pct = lm["W_pct"]
            W_pct_s = "nan" if not np.isfinite(W_pct) else f"{int(round(W_pct)):>3d}"
            slog.HYST_LADDER(
                f"  [hyst_{sid}] sigmas={sigs:<14} | "
                f"m_norm={lm['m_norm']:+.3f} | "
                f"W_pct={W_pct_s} | "
                f"dW_norm={lm['dW_norm']:+.3f} | "
                f"eff_order={lm['eff_order']:.2f} | "
                f"cross={lm['cross_rate']:.2f}/m"
            )

            # ---- appended GEOM line (same section: obeys slog.HYST_LADDER toggles)
            W_now_s = "nan" if not np.isfinite(lm.get("W_now", float("nan"))) else f"{lm['W_now']:.4f}"
            W_z_s = "nan" if not np.isfinite(lm.get("W_z", float("nan"))) else f"{lm['W_z']:+.2f}"
            W_ratio_s = "nan" if not np.isfinite(lm.get("W_ratio", float("nan"))) else f"{lm['W_ratio']:.2f}"
            ddW_s = f"{lm.get('ddW_norm', 0.0):+.3f}"
            dom_s = f"{lm.get('dom', 0.0):.2f}"
            lead_max = lm.get("leader_max_sigma")
            lead_min = lm.get("leader_min_sigma")
            leader_s = f"{lead_max}->{lead_min}" if (lead_max is not None and lead_min is not None) else "NONE"

            # cross pair breakdown (only show pairs with crossings)
            cross_pairs = lm.get("cross_pairs") or {}
            parts = []
            for k, v in cross_pairs.items():
                try:
                    c = int(v.get("count", 0))
                except Exception:
                    c = 0
                if c > 0:
                    parts.append(f"{k}:{c}")
            cross_pairs_s = ",".join(parts) if parts else "none"

            slog.HYST_LADDER(
                f"  [hyst_{sid}_GEOM] "
                f"W_now={W_now_s} | W_z={W_z_s} | W_ratio={W_ratio_s} | "
                f"ddW_norm={ddW_s} | m2_norm={lm.get('m2_norm', 0.0):+.3f} | "
                f"dom={dom_s} | leader={leader_s} | crosses={cross_pairs_s}"
            )

            # ---- added optional diagnostics
            cross_pairs_obj = lm.get("cross_pairs") or {}
            near_count = 0
            min_gap = float("inf")
            for pair_key in cross_pairs_obj.keys():
                a, b = pair_key.split("-")
                try:
                    idx_a = lm["sigmas"].index(int(a))
                    idx_b = lm["sigmas"].index(int(b))
                    vals = _safe_get(per_sigma_hist, int(a), "values", default=[])
                    vals_b = _safe_get(per_sigma_hist, int(b), "values", default=[])
                    if vals and vals_b:
                        g = abs(float(vals[-1]) - float(vals_b[-1]))
                        min_gap = min(min_gap, g)
                        if g <= max(abs(lm.get("W_now", 0.0)) * 0.08, 0.25):
                            near_count += 1
                except Exception:
                    _ = idx_a if 'idx_a' in locals() else None
                    _ = idx_b if 'idx_b' in locals() else None
            if not np.isfinite(min_gap):
                min_gap = float("nan")

            gap_vel = -float(lm.get("dW", 0.0))
            if near_count > 0 and gap_vel > 0:
                near_state = "approaching"
            elif near_count > 0 and gap_vel <= 0:
                near_state = "hovering"
            elif gap_vel > 0:
                near_state = "pressuring"
            else:
                near_state = "calm"

            leader_key = f"{sid}_leader"
            cur_leader = leader_s
            ls = (_HYST_TRACKER.get("leader_state") or {}).get(leader_key, {
                "leader": None,
                "age": 0,
                "switch_ts": [],
            })
            if ls.get("leader") == cur_leader:
                ls["age"] = int(ls.get("age", 0)) + 1
            else:
                ls["leader"] = cur_leader
                ls["age"] = 1
                ts_hist = list(ls.get("switch_ts") or [])
                ts_hist.append(decision_dt)
                ls["switch_ts"] = ts_hist[-20:]
            _HYST_TRACKER.setdefault("leader_state", {})[leader_key] = ls

            switchN = len(ls.get("switch_ts") or [])
            leader_stab = float(ls.get("age", 0) / max(1, ls.get("age", 0) + switchN))

            cross_rate = float(lm.get("cross_rate", 0.0))
            order = float(lm.get("eff_order", 0.0))
            W_stability = float(1.0 / (1.0 + abs(lm.get("ddW_norm", 0.0))))
            cross_stability = float(1.0 / (1.0 + cross_rate))
            near_penalty = min(1.0, near_count / max(1.0, float(len(cross_pairs_obj) or 1)))
            stability_score = max(0.0, min(1.0, 0.35 * cross_stability + 0.30 * order + 0.20 * W_stability + 0.15 * leader_stab - 0.20 * near_penalty))
            if stability_score < 0.25:
                stability_state = "fragile"
            elif stability_score < 0.50:
                stability_state = "unstable"
            elif stability_score < 0.75:
                stability_state = "stable"
            else:
                stability_state = "strong"

            dWn = float(lm.get("dW_norm", 0.0))
            ddWn = float(lm.get("ddW_norm", 0.0))
            if abs(dWn) < 0.05:
                spread_state = "frozen"
            elif dWn > 0:
                spread_state = "widening"
            else:
                spread_state = "narrowing"
            spread_momentum = "flat"
            if ddWn > 0.05:
                spread_momentum = "strengthening"
            elif ddWn < -0.05:
                spread_momentum = "weakening"
            exhaust = int(abs(dWn) < 0.12 and abs(ddWn) > 0.10)

            order_conf = max(0.0, min(1.0, 0.6 * order + 0.4 * cross_stability))
            order_break = max(0.0, min(1.0, 1.0 - order_conf + near_penalty * 0.4))

            risk_state = "continuation_friendly"
            if order_break > 0.70:
                risk_state = "collapse_watch"
            elif near_penalty > 0.5 and cross_rate > 0.6:
                risk_state = "whipsaw_risk"
            elif spread_state == "narrowing" and spread_momentum == "weakening":
                risk_state = "reversal_watch"

            if sec_stability:
                slog.HYST_STABILITY(
                    f"[hyst_stability] {sid} stability={stability_score:.2f} {stability_state} | "
                    f"leader_age={int(ls.get('age', 0))} switchN={switchN} leader_stab={leader_stab:.2f}"
                )
            if sec_pressure:
                slog.HYST_PRESSURE(
                    f"[hyst_pressure] {sid} near_cross={near_count} min_gap={min_gap if np.isfinite(min_gap) else float('nan'):.3f} "
                    f"gap_vel={gap_vel:+.3f} state={near_state}"
                )
            if sec_spread or sec_risk:
                slog.HYST_SPREAD_STATE(
                    f"[hyst_spread_state] {sid} spread={spread_state} momentum={spread_momentum} exhaust={exhaust} "
                    f"order_conf={order_conf:.2f} break_risk={order_break:.2f} risk={risk_state}"
                )

    if True:
        slog.HYST_DEBUG(
            f"  [hyst_dbg] tail={hyst['meta']['tail_window_seconds']}s | "
            f"baseline={hyst['meta']['baseline_window_seconds']}s | "
            f"align_tol={hyst['meta']['align_tol_seconds']:.1f}s"
        )

    slog.HYST_HEADER(_line("-", 155))

    return hyst