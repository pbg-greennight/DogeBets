from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HYST_TRACKER: Dict[str, Any] = {
    "episode_start_ts": None,
    "episode_pair": None,
    "last_primary_flip_ts": None,
    "leader_state": {},
}


def get_hyst_tracker() -> Dict[str, Any]:
    """Allow other modules to read the persistent hysteresis tracker."""
    return _HYST_TRACKER


def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


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
    if len(ts) < 5:
        return 0.0
    x = np.array([(t - ts[0]).total_seconds() for t in ts], dtype=float)
    y = vals.astype(float)
    if not np.isfinite(y).any():
        return 0.0
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
    out = np.full(len(base_ts), np.nan, dtype=float)
    if len(base_ts) == 0 or len(other_ts) == 0:
        return out

    j = 0
    for i, t in enumerate(base_ts):
        best_k = None
        best_dt = None
        while j + 1 < len(other_ts) and other_ts[j + 1] <= t:
            j += 1
        for k in (j, j + 1):
            if 0 <= k < len(other_ts):
                dt = abs((other_ts[k] - t).total_seconds())
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_k = k
        if best_k is not None and best_dt is not None and best_dt <= tol_seconds:
            out[i] = float(other_vals[best_k])
    return out


def _update_leader_tracker(stack_id: str, leader_repr: str, decision_dt: datetime) -> Dict[str, Any]:
    tracker = _HYST_TRACKER.setdefault("leader_state", {})
    state = dict(tracker.get(stack_id) or {"leader": None, "age": 0, "switch_ts": []})
    if state.get("leader") == leader_repr:
        state["age"] = int(state.get("age", 0)) + 1
    else:
        state["leader"] = leader_repr
        state["age"] = 1
        ts_hist = list(state.get("switch_ts") or [])
        ts_hist.append(decision_dt)
        state["switch_ts"] = ts_hist[-20:]
    tracker[stack_id] = state
    return state


def _derive_stack_states(
    *,
    sid: str,
    sigs: List[int],
    last_vals: np.ndarray,
    W_now: float,
    W_z: float,
    W_ratio: float,
    dW: float,
    dW_norm: float,
    ddW_norm: float,
    cross_rate: float,
    cross_pairs: Dict[str, Any],
    eff_order: float,
    leader_max_sigma: Optional[int],
    leader_min_sigma: Optional[int],
    decision_dt: datetime,
) -> Dict[str, Any]:
    finite_last = last_vals[np.isfinite(last_vals)]
    pair_gaps: List[float] = []
    near_count = 0
    threshold = max(abs(W_now) * 0.08, 0.25) if np.isfinite(W_now) else 0.25
    for i in range(len(finite_last) - 1):
        gap = abs(float(finite_last[i] - finite_last[i + 1]))
        pair_gaps.append(gap)
        if gap <= threshold:
            near_count += 1
    min_gap = float(min(pair_gaps)) if pair_gaps else float("nan")

    gap_vel = -float(dW)
    if near_count > 0 and gap_vel > 0:
        near_state = "approaching"
    elif near_count > 0 and gap_vel <= 0:
        near_state = "hovering"
    elif gap_vel > 0:
        near_state = "pressuring"
    else:
        near_state = "calm"

    leader_repr = (
        f"{leader_max_sigma}->{leader_min_sigma}"
        if leader_max_sigma is not None and leader_min_sigma is not None
        else "NONE"
    )
    leader_state = _update_leader_tracker(f"{sid}_leader", leader_repr, decision_dt)
    switch_count = len(leader_state.get("switch_ts") or [])
    leader_age = int(leader_state.get("age", 0))
    leader_stability = float(leader_age / max(1, leader_age + switch_count))

    cross_stability = float(1.0 / (1.0 + max(cross_rate, 0.0)))
    W_stability = float(1.0 / (1.0 + abs(ddW_norm)))
    near_penalty = min(1.0, near_count / max(1.0, float(len(cross_pairs) or 1.0)))
    stability_score = _clamp(
        0.35 * cross_stability
        + 0.30 * float(eff_order)
        + 0.20 * W_stability
        + 0.15 * leader_stability
        - 0.20 * near_penalty
    )
    if stability_score < 0.25:
        stability_state = "fragile"
    elif stability_score < 0.50:
        stability_state = "unstable"
    elif stability_score < 0.75:
        stability_state = "stable"
    else:
        stability_state = "strong"

    if abs(dW_norm) < 0.05:
        spread_state = "frozen"
    elif dW_norm > 0:
        spread_state = "widening"
    else:
        spread_state = "narrowing"

    if ddW_norm > 0.05:
        spread_momentum = "strengthening"
    elif ddW_norm < -0.05:
        spread_momentum = "weakening"
    else:
        spread_momentum = "flat"

    exhaust = int(abs(dW_norm) < 0.12 and abs(ddW_norm) > 0.10)
    order_conf = _clamp(0.6 * float(eff_order) + 0.4 * cross_stability)
    break_risk = _clamp(1.0 - order_conf + near_penalty * 0.4)

    risk_state = "continuation_friendly"
    if break_risk > 0.70:
        risk_state = "collapse_watch"
    elif near_penalty > 0.5 and cross_rate > 0.6:
        risk_state = "whipsaw_risk"
    elif spread_state == "narrowing" and spread_momentum == "weakening":
        risk_state = "reversal_watch"

    fan_tightness = _clamp(1.0 / (1.0 + max(W_ratio, 0.0))) if np.isfinite(W_ratio) else 0.0
    stack_alignment = _clamp(float(eff_order))
    ladder_monotonic = 1.0 if float(eff_order) >= 0.95 else 0.0
    ladder_compression = _clamp(1.0 - min(1.0, max(W_ratio, 0.0))) if np.isfinite(W_ratio) else 0.0

    return {
        "state": spread_state,
        "pressure": near_state,
        "risk": risk_state,
        "stability": stability_score,
        "stability_state": stability_state,
        "near_cross": int(near_count),
        "near_cross_state": near_state,
        "min_gap": min_gap,
        "gap_vel": gap_vel,
        "spread_state": spread_state,
        "spread_momentum": spread_momentum,
        "spread_slope": float(dW_norm),
        "spread_accel": float(ddW_norm),
        "exhaust": int(exhaust),
        "order_conf": order_conf,
        "break_risk": break_risk,
        "fan_tightness": fan_tightness,
        "stack_alignment": stack_alignment,
        "leader_age": leader_age,
        "switch_count": int(switch_count),
        "leader_stability": leader_stability,
        "ladder_monotonic": ladder_monotonic,
        "ladder_compression": ladder_compression,
    }


def _safe_seconds_between(later: Any, earlier: Any) -> Optional[float]:
    try:
        if later is None or earlier is None:
            return None
        if isinstance(later, datetime) and isinstance(earlier, datetime):
            return float((later - earlier).total_seconds())
    except Exception:
        return None
    return None


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

    Phase 2 persists the spread/risk/stability/pressure states in the returned
    object so both the live runner and the printer consume the same analysis.
    """

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

    ts83, g83 = _series(83)
    cutoff = decision_dt - timedelta(seconds=lookback_seconds)
    mask_base = np.array([t >= cutoff for t in ts83], dtype=bool)

    mode = "DEFAULT"
    pair = primary_default

    def _d_pair(a: int, b: int) -> Tuple[np.ndarray, Optional[datetime]]:
        ts_a, g_a = _series(a)
        g_a_al = _nearest_align(ts83, ts_a, g_a, tol_seconds=align_tol_seconds)
        d = g_a_al - g83
        last_flip_full = _find_last_sign_flip(ts83, d)
        return d, last_flip_full

    d38_83, last_flip_38 = _d_pair(38, 83)
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

    start_ts = last_cross if (last_cross is not None and last_cross >= cutoff) else None
    elapsed = (decision_dt - start_ts).total_seconds() if start_ts else None
    stage = "IN_PROGRESS"

    ts23, g23 = _series(probe_pair[0])
    g23_al = _nearest_align(ts83, ts23, g23, tol_seconds=align_tol_seconds)
    d23_83 = g23_al - g83
    probe_last_flip = _find_last_sign_flip(ts83, d23_83)

    flip_watch = bool(start_ts and probe_last_flip and probe_last_flip > start_ts)
    tail_cut = decision_dt - timedelta(seconds=tail_seconds)
    mask_tail = np.array([t >= tail_cut for t in ts83], dtype=bool)
    ts_tail = [ts83[i] for i in range(len(ts83)) if mask_tail[i]]
    dprobe_tail = np.abs(d23_83[mask_tail])
    dprobe_slope = _compute_slope(ts_tail, dprobe_tail)
    fast_collapse = bool(dprobe_slope < 0)
    probe_recovery = bool(dprobe_slope > 0)

    ladder_defs = {
        "S0": [23, 38, 53, 68, 83],
        "S1": [38, 53, 68, 83],
        "S2": [53, 68, 83],
        "S3": [68, 83],
    }

    stacks: Dict[str, Any] = {}
    for sid, sigs in ladder_defs.items():
        series_map: Dict[int, np.ndarray] = {}
        for s in sigs:
            ts_s, g_s = _series(s)
            series_map[s] = _nearest_align(ts83, ts_s, g_s, tol_seconds=align_tol_seconds)

        vals_stack = np.vstack([series_map[s] for s in sigs])
        mean_curve = np.nanmean(vals_stack, axis=0)

        mean_tail = mean_curve[mask_tail]
        m = _compute_slope(ts_tail, mean_tail)
        m2 = _compute_accel(ts_tail, mean_tail)

        mean_base = mean_curve[mask_base]
        m_scale = _mad(mean_base)
        m_norm = float(m / (m_scale + 1e-12)) if np.isfinite(m_scale) else 0.0
        m2_norm = float(m2 / (m_scale + 1e-12)) if np.isfinite(m_scale) else 0.0

        spread_series = np.nanmax(vals_stack, axis=0) - np.nanmin(vals_stack, axis=0)
        W_now = float(spread_series[-1]) if np.isfinite(spread_series[-1]) else float("nan")
        spread_base = spread_series[mask_base]
        W_p25 = _percentile(spread_base, 25)
        W_p75 = _percentile(spread_base, 75)
        W_med = float(np.median(spread_base)) if spread_base.size else 0.0
        W_mad = _mad(spread_base)
        W_z = float((W_now - W_med) / (W_mad + 1e-12)) if np.isfinite(W_now) else float("nan")
        W_ratio = float(W_now / (W_p75 + 1e-12)) if np.isfinite(W_now) else float("nan")
        if spread_base.size > 10 and np.isfinite(W_now):
            W_pct = float(np.mean(spread_base <= W_now) * 100.0)
        else:
            W_pct = float("nan")

        spread_tail = spread_series[mask_tail]
        dW = _compute_slope(ts_tail, spread_tail)
        ddW = _compute_accel(ts_tail, spread_tail)
        dW_scale = _mad(spread_base)
        dW_norm = float(dW / (dW_scale + 1e-12)) if np.isfinite(dW_scale) else 0.0
        ddW_norm = float(ddW / (dW_scale + 1e-12)) if np.isfinite(dW_scale) else 0.0

        last_vals = vals_stack[:, -1]
        if np.all(np.isfinite(last_vals)):
            bull_ok = np.all(np.diff(last_vals) < 0)
            bear_ok = np.all(np.diff(last_vals) > 0)
            order_bull = 1.0 if bull_ok else 0.0
            order_bear = 1.0 if bear_ok else 0.0
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

        cross_count = 0
        cross_pairs: Dict[str, Any] = {}
        pairs = list(zip(sigs[:-1], sigs[1:]))
        for a, b in pairs:
            da = series_map[a][mask_tail] - series_map[b][mask_tail]
            sgn = np.array([_sign(x) for x in da], dtype=int)
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

        order_stability = float(1.0 / (1.0 + cross_rate))
        eff_order = float(max(order_bull, order_bear) * order_stability)
        eff_order_ok = None

        derived = _derive_stack_states(
            sid=sid,
            sigs=sigs,
            last_vals=last_vals,
            W_now=W_now,
            W_z=W_z,
            W_ratio=W_ratio,
            dW=dW,
            dW_norm=dW_norm,
            ddW_norm=ddW_norm,
            cross_rate=cross_rate,
            cross_pairs=cross_pairs,
            eff_order=eff_order,
            leader_max_sigma=leader_max_sigma,
            leader_min_sigma=leader_min_sigma,
            decision_dt=decision_dt,
        )

        stack_obj = {
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
        stack_obj.update(derived)
        stacks[sid] = stack_obj

    eta_to_end_seconds = None
    eta_source = None
    s1 = stacks.get("S1")
    if s1 and np.isfinite(s1["W_now"]) and s1["dW"] < 0:
        eta_to_end_seconds = float(abs(s1["W_now"] / max(abs(s1["dW"]), 1e-12)))
        eta_source = "S1"

    summary_stack_id = "S1" if "S1" in stacks else (next(iter(stacks.keys())) if stacks else None)
    summary_stack = stacks.get(summary_stack_id or "", {}) or {}
    spread_state = {
        "stack": summary_stack_id,
        "spread": summary_stack.get("spread_state"),
        "momentum": summary_stack.get("spread_momentum"),
        "risk": summary_stack.get("risk"),
        "pressure": summary_stack.get("pressure"),
        "stability": summary_stack.get("stability"),
        "stability_state": summary_stack.get("stability_state"),
        "near_cross_state": summary_stack.get("near_cross_state"),
        "fan_tightness": summary_stack.get("fan_tightness"),
        "stack_alignment": summary_stack.get("stack_alignment"),
    }

    hyst_obj = {
        "method_id": "method_hyster_v1.1",
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
            "recovery": bool(probe_recovery),
            "d_abs_slope_tail": float(dprobe_slope),
            "last_flip_ts": probe_last_flip,
        },
        "spread_state": spread_state,
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
            "summary_stack": summary_stack_id,
            "leader_max_sigma": summary_stack.get("leader_max_sigma"),
            "leader_min_sigma": summary_stack.get("leader_min_sigma"),
            "cross_rate": summary_stack.get("cross_rate"),
            "order_stability": summary_stack.get("order_stability"),
            "ladder_monotonic": summary_stack.get("ladder_monotonic"),
            "ladder_compression": summary_stack.get("ladder_compression"),
        },
    }

    return hyst_obj


def flatten_hysteresis_to_src(
    hyst_obj: dict,
    config: Optional[dict] = None,
) -> dict:
    """Flatten live hysteresis payload into canonical src_hyst_* columns."""
    out: Dict[str, Any] = {}

    episode = hyst_obj.get("episode", {}) or {}
    probe = hyst_obj.get("probe", {}) or {}
    spread = hyst_obj.get("spread_state", {}) or {}
    eta = hyst_obj.get("eta", {}) or {}
    stacks = hyst_obj.get("stacks", {}) or {}
    meta = hyst_obj.get("meta", {}) or {}

    decision_time = meta.get("decision_time")
    last_cross_ts = episode.get("last_cross_ts")

    out["src_hyst_primary_sign"] = episode.get("sign_now")
    out["src_hyst_episode_age_sec"] = episode.get("elapsed_seconds")
    out["src_hyst_last_cross_age_sec"] = _safe_seconds_between(decision_time, last_cross_ts)

    out["src_hyst_probe_sign"] = probe.get("sign_now")
    out["src_hyst_probe_flip_watch"] = probe.get("flip_watch")
    out["src_hyst_probe_fast_collapse"] = probe.get("fast_collapse")
    out["src_hyst_probe_recovery"] = probe.get("recovery")
    out["src_hyst_fast_collapse"] = probe.get("fast_collapse")  # compatibility alias
    out["src_hyst_eta_to_end_seconds"] = eta.get("eta_to_end_seconds")

    out["src_hyst_summary_stack"] = spread.get("stack") or meta.get("summary_stack")
    out["src_hyst_spread_state"] = spread.get("spread")
    out["src_hyst_spread_risk"] = spread.get("risk")
    out["src_hyst_pressure_state"] = spread.get("pressure")
    out["src_hyst_stack_stability"] = spread.get("stability")
    out["src_hyst_stack_stability_state"] = spread.get("stability_state")
    out["src_hyst_near_cross_state"] = spread.get("near_cross_state")
    out["src_hyst_fan_tightness"] = spread.get("fan_tightness")
    out["src_hyst_stack_alignment"] = spread.get("stack_alignment")

    for stack_name in ["S0", "S1", "S2", "S3"]:
        s = stacks.get(stack_name, {}) or {}
        base = f"src_hyst_{stack_name.lower()}"
        out[f"{base}_state"] = s.get("state")
        out[f"{base}_pressure"] = s.get("pressure")
        out[f"{base}_risk"] = s.get("risk")
        out[f"{base}_stability"] = s.get("stability")
        out[f"{base}_stability_state"] = s.get("stability_state")
        out[f"{base}_near_cross"] = s.get("near_cross")
        out[f"{base}_near_cross_state"] = s.get("near_cross_state")
        out[f"{base}_spread_state"] = s.get("spread_state")
        out[f"{base}_spread_momentum"] = s.get("spread_momentum")
        out[f"{base}_spread_slope"] = s.get("spread_slope")
        out[f"{base}_spread_accel"] = s.get("spread_accel")
        out[f"{base}_break_risk"] = s.get("break_risk")
        out[f"{base}_fan_tightness"] = s.get("fan_tightness")
        out[f"{base}_stack_alignment"] = s.get("stack_alignment")
        out[f"{base}_leader_age"] = s.get("leader_age")
        out[f"{base}_switch_count"] = s.get("switch_count")

    out["src_hyst_leader_max_sigma"] = meta.get("leader_max_sigma")
    out["src_hyst_leader_min_sigma"] = meta.get("leader_min_sigma")
    out["src_hyst_cross_rate"] = meta.get("cross_rate")
    out["src_hyst_order_stability"] = meta.get("order_stability")
    out["src_hyst_ladder_monotonic"] = meta.get("ladder_monotonic")
    out["src_hyst_ladder_compression"] = meta.get("ladder_compression")

    return out
