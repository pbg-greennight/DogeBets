# main/engine/process/printing/DB_process_printing_hyst.py

import math
import statistics
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from .DB_process_printing_utils import (
    _safe_get,
    _fmt_time,
    _line,
    _print_cfg,
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


def _tail_preview(label: str, arr: np.ndarray, n: int = 10) -> str:
    """Debug helper to show last n values of a diff series."""
    if arr.size == 0:
        return f"{label}=<empty>"
    tail = arr[-n:] if arr.size >= n else arr
    return f"{label}=[" + ", ".join(f"{x:+.6f}" for x in tail) + "]"


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
    per_sigma_hist: dict,
    decision_dt,
    config: dict | None = None,
    logger=None,
    lookback_seconds: int | None = None,
    tail_seconds: int | None = None,
    align_tol_seconds: float | None = None,
    primary_default: tuple[int, int] | list[int] | None = None,
    primary_slow: tuple[int, int] | list[int] | None = None,
    probe_pair: tuple[int, int] | list[int] | None = None,
    **kwargs,
) -> dict:
    """
    Compute hysteresis features used by:
      - print_hysteresis_fan_stack()
      - trend_method_v2 (DISORDER / ORDER checks)
    This function is intentionally defensive: it always returns a dict with
    stable keys, even when inputs/config are missing.

    Expected per_sigma_hist:
        {sigma: {"values": [...], "times": [...]}} OR {sigma: {"values": [...]}}
        times are optional; when absent we won't compute cross timestamps.

    Config (optional):
        time_inputs_pairs: {"pairs": {"episode_primary_default":[38,83], ...}}
        hysteresis_engine: ... (not required here)
    """
    # Backward-compatible aliases used by older callers.
    if lookback_seconds is None and "lookback" in kwargs:
        lookback_seconds = kwargs.pop("lookback")
    if tail_seconds is None and "tail" in kwargs:
        tail_seconds = kwargs.pop("tail")
    if align_tol_seconds is None and "align_tol" in kwargs:
        align_tol_seconds = kwargs.pop("align_tol")
    if primary_default is None and "primary_pair" in kwargs:
        primary_default = kwargs.pop("primary_pair")
    if probe_pair is None and "probe_default" in kwargs:
        probe_pair = kwargs.pop("probe_default")
    if kwargs:
        bad = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"compute_hysteresis_features() got unexpected keyword argument(s): {bad}")

    # Currently not used by this lightweight feature builder, but accepted so callers can
    # pass the same config block consistently.
    _ = (lookback_seconds, tail_seconds, align_tol_seconds, primary_slow)

    cfg = config or {}
    pairs_cfg = (cfg.get("time_inputs_pairs") or {}).get("pairs") or {}

    def _get_series(sigma: int) -> list[float]:
        blob = per_sigma_hist.get(sigma) or {}
        vals = blob.get("values") if isinstance(blob, dict) else None
        if vals is None and isinstance(blob, (list, tuple)):
            vals = list(blob)
        return list(vals or [])

    def _diff_series(a: int, b: int) -> list[float]:
        va = _get_series(a)
        vb = _get_series(b)
        n = min(len(va), len(vb))
        if n <= 0:
            return []
        return [va[-n + i] - vb[-n + i] for i in range(n)]

    def _sign(x: float) -> int:
        return 1 if x > 0 else (-1 if x < 0 else 0)

    def _safe_mad(xs: list[float]) -> float:
        if not xs:
            return 0.0
        m = statistics.median(xs)
        dev = [abs(x - m) for x in xs]
        return statistics.median(dev)

    def _tail_slope(xs: list[float], tail: int = 60) -> float:
        if not xs:
            return 0.0
        seg = xs[-min(tail, len(xs)):]
        if len(seg) < 2:
            return 0.0
        # simple slope per-sample
        return (seg[-1] - seg[0]) / float(len(seg) - 1)

    def _cross_rate(xs: list[float], tail: int = 300) -> float:
        if not xs:
            return 0.0
        seg = xs[-min(tail, len(xs)):]
        if len(seg) < 3:
            return 0.0
        s_prev = _sign(seg[0])
        flips = 0
        steps = 0
        for v in seg[1:]:
            s = _sign(v)
            if s == 0:
                continue
            if s_prev == 0:
                s_prev = s
                continue
            steps += 1
            if s != s_prev:
                flips += 1
                s_prev = s
        return (flips / steps) if steps else 0.0

    def _percentile_rank(abs_now: float, abs_series: list[float]) -> float:
        if not abs_series:
            return 0.0
        less = sum(1 for x in abs_series if x <= abs_now)
        return less / float(len(abs_series))

    # --- Choose primary + probe pairs (default to (38,83) and (23,83)) ---
    primary_pair = primary_default or pairs_cfg.get("episode_primary_default") or [38, 83]
    probe_pair = probe_pair or pairs_cfg.get("episode_probe_default") or [23, 83]

    try:
        p_a, p_b = int(primary_pair[0]), int(primary_pair[1])
    except Exception:
        p_a, p_b = 38, 83
    try:
        q_a, q_b = int(probe_pair[0]), int(probe_pair[1])
    except Exception:
        q_a, q_b = 23, 83

    d_primary = _diff_series(p_a, p_b)
    d_probe = _diff_series(q_a, q_b)

    # Fallback if series missing: return minimal structure to avoid KeyErrors downstream
    if not d_primary:
        return {
            "pair_used": [p_a, p_b],
            "eta": None,
            "episode": {"pair_used": [p_a, p_b]},
            "primary": {"pair": [p_a, p_b], "now": None, "sign": 0, "last_cross": None},
            "probe": {"pair": [q_a, q_b], "now": None, "sign": 0, "last_cross": None},
            "stacks": {"S1": {"stacks": {}, "note": "missing primary diff series"}},
        }

    now_p = float(d_primary[-1])
    now_q = float(d_probe[-1]) if d_probe else 0.0

    # ETA to cross (very rough): time to reach 0 given tail slope
    slope_p = _tail_slope(d_primary, tail=60)
    eta = None
    if slope_p != 0.0:
        eta_samp = (-now_p) / slope_p
        if eta_samp > 0:
            # we don't know seconds/sample reliably here; treat as samples
            eta = float(eta_samp)

    abs_p = [abs(x) for x in d_primary]
    mad_p = _safe_mad(abs_p) or 1e-12
    W_now = abs(now_p) / mad_p
    W_p75 = (sorted(abs_p)[int(0.75 * (len(abs_p) - 1))] / mad_p) if len(abs_p) >= 2 else W_now
    W_pct = _percentile_rank(abs(now_p), abs_p)

    cross_rate = _cross_rate(d_primary, tail=300)
    eff_order = max(0.0, 1.0 - cross_rate)

    # "order" scores: simplistic but stable
    order_bull = 1.0 if now_p > 0 else 0.0
    order_bear = 1.0 if now_p < 0 else 0.0

    # normalized slope magnitude
    m_norm = abs(slope_p) / mad_p

    # Build S1 summary expected by trend_method_v2
    s1 = {
        "pair_used": [p_a, p_b],
        "probe_pair": [q_a, q_b],
        "d_now": now_p,
        "probe_now": now_q,
        "cross_rate": cross_rate,
        "eff_order": eff_order,
        "order_bull": order_bull,
        "order_bear": order_bear,
        "W_now": W_now,
        "W_p75": W_p75,
        "W_pct": W_pct,
        "m_norm": m_norm,
        "eta_samples": eta,
    }

    return {
        "pair_used": [p_a, p_b],
        "eta": eta,
        "episode": {
            "pair_used": [p_a, p_b],
            "pair_probe": [q_a, q_b],
        },
        "primary": {
            "pair": [p_a, p_b],
            "now": now_p,
            "sign": _sign(now_p),
            "d_abs_slope_tail": abs(slope_p),
        },
        "probe": {
            "pair": [q_a, q_b],
            "now": now_q,
            "sign": _sign(now_q),
        },
        "stacks": {"S1": {"stacks": s1}},
    }
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

    if config is None:
        config = {}

    p = _print_cfg(config)
    hcfg = p.get("HYSTERESIS", {}) if isinstance(p.get("HYSTERESIS", {}), dict) else {}
    enabled = bool(hcfg.get("ENABLED", True))
    # Legacy fallback
    if not enabled and not bool(config.get("LOG_HYSTERESIS", True)):
        return {}

    on_header = bool((hcfg.get("HEADER", {}) if isinstance(hcfg.get("HEADER", {}), dict) else {}).get("ENABLED", True))
    on_episode = bool((hcfg.get("EPISODE", {}) if isinstance(hcfg.get("EPISODE", {}), dict) else {}).get("ENABLED", True))
    on_probe = bool((hcfg.get("PROBE", {}) if isinstance(hcfg.get("PROBE", {}), dict) else {}).get("ENABLED", True))
    on_eta = bool((hcfg.get("ETA", {}) if isinstance(hcfg.get("ETA", {}), dict) else {}).get("ENABLED", True))
    on_ladder = bool((hcfg.get("LADDER", {}) if isinstance(hcfg.get("LADDER", {}), dict) else {}).get("ENABLED", True))
    on_debug = bool((hcfg.get("DEBUG", {}) if isinstance(hcfg.get("DEBUG", {}), dict) else {}).get("ENABLED", False))

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
    ep = hyst.get("episode") or {}
    pr = hyst.get("probe") or {}

    # Some builders return ETA as scalar/None while others return a dict.
    # Normalize so log formatting never throws.
    raw_eta = hyst.get("eta", {})
    if isinstance(raw_eta, dict):
        eta = raw_eta
    elif raw_eta is None:
        eta = {}
    else:
        try:
            eta_val = float(raw_eta)
        except Exception:
            eta = {}
        else:
            eta = {"eta_to_end_seconds": eta_val, "source_stack": "S1"}

    stacks = hyst.get("stacks") or {}

    if on_header:
        slog.HYST_HEADER(_line("=", 155))
        slog.HYST_HEADER("HYSTERESIS FAN STACK (method_hyster_v1.0)")
        slog.HYST_HEADER(_line("-", 155))

    if on_episode:
        ep_pair = ep.get("pair_used") or [None, None]
        ep_pair_a = ep_pair[0] if len(ep_pair) > 0 else None
        ep_pair_b = ep_pair[1] if len(ep_pair) > 1 else None
        # Older/newer feature payloads may expose start timestamp under different keys.
        ep_start_ts = ep.get("start_ts") if isinstance(ep, dict) else None
        if ep_start_ts is None and isinstance(ep, dict):
            ep_start_ts = ep.get("episode_start_ts")
        ep_elapsed = ep.get("elapsed_seconds")
        ep_stage = ep.get("stage") or "UNKNOWN"

        slog.HYST_EPISODE(
            f"[hyst_episode] decision_time={_fmt_time(decision_dt)} | "
            f"pair=({ep_pair_a},{ep_pair_b}) | "
            f"start={_fmt_time(ep_start_ts) if ep_start_ts else 'NONE'} | "
            f"elapsed={(ep_elapsed if ep_elapsed is not None else 0.0):.1f}s | "
            f"stage={ep_stage}"
        )

        ep_d_now = float(ep.get("d_now") or 0.0)
        ep_sign_now = int(ep.get("sign_now") or 0)
        ep_last_cross = ep.get("last_cross_ts")
        ep_mode = ep.get("mode") or "unknown"
        slog.HYST_EPISODE(
            f"[hyst_primary] d{ep_pair_a}_{ep_pair_b} now={ep_d_now:+.6f} | "
            f"sign={ep_sign_now} | "
            f"last_cross={_fmt_time(ep_last_cross) if ep_last_cross else 'NONE'} | "
            f"lookback=60m | mode={ep_mode}"
        )

    if on_probe:
        slog.HYST_PROBE(
            f"[hyst_probe] pair=(23,83) | d23_83 now={float(pr.get('d_now') or 0.0):+.6f} | "
            f"sign={int(pr.get('sign_now') or 0)} | flip_watch={int(pr.get('flip_watch') or 0)} | "
            f"fast_collapse={int(pr.get('fast_collapse') or 0)} | "
            f"d_abs_slope_tail={float(pr.get('d_abs_slope_tail') or 0.0):+.6f}"
        )

    if eta.get("eta_to_end_seconds") is not None:
        slog.HYST_ETA(
            f"[hyst_eta] eta={eta['eta_to_end_seconds']:.1f}s (~{eta['eta_to_end_seconds'] / 300.0:.2f} epochs) | "
            f"source={eta.get('source_stack')}"
        )

    if on_ladder:
        for sid in ["S0", "S1", "S2", "S3"]:
            lm = stacks.get(sid)
            if not lm:
                continue
            sigmas = lm.get("sigmas") if isinstance(lm, dict) else None
            sigs = ",".join(map(str, sigmas or []))
            W_pct = float(lm.get("W_pct", float("nan"))) if isinstance(lm, dict) else float("nan")
            W_pct_s = "nan" if not np.isfinite(W_pct) else f"{int(round(W_pct)):>3d}"
            slog.HYST_LADDER(
                f"  [hyst_{sid}] sigmas={sigs:<14} | "
                f"m_norm={float(lm.get('m_norm', 0.0)):+.3f} | "
                f"W_pct={W_pct_s} | "
                f"dW_norm={float(lm.get('dW_norm', 0.0)):+.3f} | "
                f"eff_order={float(lm.get('eff_order', 0.0)):.2f} | "
                f"cross={float(lm.get('cross_rate', 0.0)):.2f}/m"
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

    if on_debug:
        slog.HYST_DEBUG(
            f"  [hyst_dbg] tail={hyst['meta']['tail_window_seconds']}s | "
            f"baseline={hyst['meta']['baseline_window_seconds']}s | "
            f"align_tol={hyst['meta']['align_tol_seconds']:.1f}s"
        )

    slog.HYST_HEADER(_line("-", 155))

    return hyst
