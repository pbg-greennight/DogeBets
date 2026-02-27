"""main/engine/process/DB_process_metrics.py

Snapshot metrics and formatting helpers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import math


def _to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None



def classify_direction(slope: float, curve: float, eps: float = 1e-12) -> str:
    """Classify movement tag based on slope (1st derivative proxy) and curve (2nd derivative proxy).

    Convention used by your log:
      - slope > 0 : UP
      - slope < 0 : DOWN
      - curve > 0 : accelerating in the *direction of slope*
      - curve < 0 : decelerating / bending against the direction
    """
    if abs(slope) <= eps:
        # If we're basically flat, use curvature to hint bias, else FLAT
        if abs(curve) <= eps:
            return "FLAT"
        return "FLAT/accel" if curve > 0 else "FLAT/decel"

    if slope > 0:
        return "UP/accel" if curve > eps else "UP/decel"
    else:
        # slope < 0
        return "DOWN/accel" if curve > eps else "DOWN/decel"

def snapshot_metrics(values: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
    """Compute metrics for a 1D series segment.

    Existing keys (kept for compatibility):
      last, delta, slope, curve, tag

    Extra diagnostics (for later rule-building & offline analysis):
      n
      lin_slope, lin_r2
      quad_a, quad_b, quad_r2, quad_tangent, quad_curv
      z_last
      slope_mean, slope_std
      run_score
    """
    n = len(values)
    if n == 0:
        return {"n": 0}

    if n == 1:
        v = float(values[-1])
        return {
            "n": 1,
            "start": v,
            "last": v,
            "delta": 0.0,
            "slope": 0.0,
            "curve": 0.0,
            "tag": "flat",
            "lin_slope": 0.0,
            "lin_r2": 0.0,
            "quad_a": 0.0,
            "quad_b": 0.0,
            "quad_r2": 0.0,
            "quad_tangent": 0.0,
            "quad_curv": 0.0,
            "z_last": 0.0,
            "slope_mean": 0.0,
            "slope_std": 0.0,
            "run_score": 0.0,
        }

    # ---- base slope/curve used in logs ----
    start_val = float(values[0])
    end_val = float(values[-1])
    delta = end_val - start_val

    dt = max((timestamps[-1] - timestamps[0]).total_seconds(), 1e-9)
    slope = delta / dt

    # curvature proxy: late slope - early slope (using midpoint)
    mid_idx = n // 2
    mid_val = float(values[mid_idx])
    mid_dt = max((timestamps[mid_idx] - timestamps[0]).total_seconds(), 1e-9)
    tail_dt = max((timestamps[-1] - timestamps[mid_idx]).total_seconds(), 1e-9)
    slope1 = (mid_val - start_val) / mid_dt
    slope2 = (end_val - mid_val) / tail_dt
    curve = slope2 - slope1

    # ---- first-difference stats (run stability) ----
    diffs = []
    for i in range(1, n):
        dt_i = (timestamps[i] - timestamps[i - 1]).total_seconds()
        if dt_i <= 0:
            continue
        diffs.append((float(values[i]) - float(values[i - 1])) / dt_i)

    if diffs:
        slope_mean = sum(diffs) / len(diffs)
        mu_d = slope_mean
        slope_std = math.sqrt(sum((d - mu_d) ** 2 for d in diffs) / len(diffs))
    else:
        slope_mean = 0.0
        slope_std = 0.0

    # ---- linear regression (least squares) ----
    xs = [(t - timestamps[0]).total_seconds() for t in timestamps]
    x_mean = sum(xs) / n
    y_mean = sum(float(v) for v in values) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    sxy = sum((x - x_mean) * (float(y) - y_mean) for x, y in zip(xs, values))

    if sxx > 0:
        lin_slope = sxy / sxx
        lin_intercept = y_mean - lin_slope * x_mean
        ss_tot = sum((float(y) - y_mean) ** 2 for y in values)
        ss_res = sum((float(y) - (lin_intercept + lin_slope * x)) ** 2 for x, y in zip(xs, values))
        lin_r2 = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - (ss_res / ss_tot))
    else:
        lin_slope = 0.0
        lin_r2 = 0.0

    # ---- quadratic regression (least squares) ----
    quad_a = quad_b = quad_c = 0.0
    quad_r2 = 0.0
    quad_tangent = 0.0
    quad_curv = 0.0

    if n >= 3 and len(set(xs)) >= 3:
        Sx = sum(xs)
        Sx2 = sum(x * x for x in xs)
        Sx3 = sum(x ** 3 for x in xs)
        Sx4 = sum(x ** 4 for x in xs)
        Sy = sum(float(y) for y in values)
        Sxy = sum(x * float(y) for x, y in zip(xs, values))
        Sx2y = sum((x * x) * float(y) for x, y in zip(xs, values))

        def det3(a11, a12, a13, a21, a22, a23, a31, a32, a33):
            return (
                a11 * (a22 * a33 - a23 * a32)
                - a12 * (a21 * a33 - a23 * a31)
                + a13 * (a21 * a32 - a22 * a31)
            )

        D = det3(Sx4, Sx3, Sx2, Sx3, Sx2, Sx, Sx2, Sx, n)
        if abs(D) > 1e-12:
            Da = det3(Sx2y, Sx3, Sx2, Sxy, Sx2, Sx, Sy, Sx, n)
            Db = det3(Sx4, Sx2y, Sx2, Sx3, Sxy, Sx, Sx2, Sy, n)
            Dc = det3(Sx4, Sx3, Sx2y, Sx3, Sx2, Sxy, Sx2, Sx, Sy)
            quad_a = Da / D
            quad_b = Db / D
            quad_c = Dc / D

            x_last = xs[-1]
            quad_tangent = 2.0 * quad_a * x_last + quad_b
            quad_curv = 2.0 * quad_a

            y_mean2 = y_mean
            ss_tot2 = sum((float(y) - y_mean2) ** 2 for y in values)
            ss_res2 = sum((float(y) - (quad_a * x * x + quad_b * x + quad_c)) ** 2 for x, y in zip(xs, values))
            quad_r2 = 0.0 if ss_tot2 <= 0 else max(0.0, 1.0 - (ss_res2 / ss_tot2))

    # ---- z-score of last within segment ----
    mu = y_mean
    var = sum((float(v) - mu) ** 2 for v in values) / n
    sd = math.sqrt(var) if var > 0 else 0.0
    z_last = 0.0 if sd == 0 else (end_val - mu) / sd

    # ---- run strength score (unitless) ----
    run_score = (abs(lin_slope) * (0.5 + 0.5 * lin_r2)) / (1.0 + slope_std)

    tag = classify_direction(slope, curve)

    return {
        "n": n,
        "start": start_val,
        "last": end_val,
        "delta": delta,
        "slope": slope,
        "curve": curve,
        "tag": tag,
        "lin_slope": lin_slope,
        "lin_r2": lin_r2,
        "quad_a": quad_a,
        "quad_b": quad_b,
        "quad_r2": quad_r2,
        "quad_tangent": quad_tangent,
        "quad_curv": quad_curv,
        "z_last": z_last,
        "slope_mean": slope_mean,
        "slope_std": slope_std,
        "run_score": run_score,
    }


def _fmt_num(x: Any) -> str:
    fx = _to_float_or_none(x)
    if fx is None:
        return "None"
    return f"{fx:.2f}"


def _fmt_metric(x: Any) -> str:
    fx = _to_float_or_none(x)
    if fx is None:
        return "None"
    return f"{fx:.6f}"


def format_series_values(values: List[Any]) -> str:
    return ", ".join(_fmt_num(v) for v in values)


# ---------------------------------------------------------------------
# Bell-curve (Peak/Valley) helpers
# ---------------------------------------------------------------------

def _is_valid_number(v: Any) -> bool:
    try:
        if v is None:
            return False
        fv = float(v)
        return fv == fv  # not NaN
    except Exception:
        return False


def find_last_extrema_pair(
    ts_list: List[datetime],
    values_list: List[Any],
    end_dt: datetime,
    min_sep_seconds: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """Find the most recent alternating extrema pair (prev, last) up to end_dt.

    Returns:
      {
        "prev": {"idx": int, "ts": datetime, "val": float, "kind": "PEAK"|"VALLEY"},
        "last": {"idx": int, "ts": datetime, "val": float, "kind": "PEAK"|"VALLEY"},
        "swing": "VALLEY→PEAK" | "PEAK→VALLEY"
      }

    Notes:
      - Uses a simple 3-point local-extrema test on the reference series.
      - Skips invalid/None/NaN values.
      - Enforces a minimum time separation between the two extrema.
      - Requires the two extrema to be opposite kinds (alternating).
    """

    n = min(len(ts_list), len(values_list))
    if n < 3:
        return None

    ts = ts_list[:n]
    vals_raw = values_list[:n]

    # Consider only points <= end_dt
    eligible_idxs = [i for i, t in enumerate(ts) if t <= end_dt]
    if len(eligible_idxs) < 3:
        return None
    lo = eligible_idxs[0]
    hi = eligible_idxs[-1]

    def _val(i: int) -> Optional[float]:
        v = vals_raw[i]
        if not _is_valid_number(v):
            return None
        return float(v)

    def _kind(i: int) -> Optional[str]:
        # Local extrema using i-1, i, i+1
        if i - 1 < lo or i + 1 > hi:
            return None
        v0 = _val(i - 1)
        v1 = _val(i)
        v2 = _val(i + 1)
        if v0 is None or v1 is None or v2 is None:
            return None

        if v0 < v1 > v2:
            return "PEAK"
        if v0 > v1 < v2:
            return "VALLEY"
        return None

    # Walk backwards to find the most recent extremum, then the previous opposite extremum.
    last = None
    for i in range(hi - 1, lo + 0, -1):
        k = _kind(i)
        if k is None:
            continue
        v = _val(i)
        if v is None:
            continue
        last = {"idx": i, "ts": ts[i], "val": v, "kind": k}
        break

    if last is None:
        return None

    prev = None
    for j in range(last["idx"] - 1, lo + 0, -1):
        k = _kind(j)
        if k is None:
            continue
        if k == last["kind"]:
            continue  # require alternation
        v = _val(j)
        if v is None:
            continue
        if (last["ts"] - ts[j]).total_seconds() < float(min_sep_seconds):
            continue
        prev = {"idx": j, "ts": ts[j], "val": v, "kind": k}
        break

    if prev is None:
        return None

    swing = f"{prev['kind']}→{last['kind']}"
    swing = swing.replace("VALLEY", "VALLEY").replace("PEAK", "PEAK")
    # Pretty arrow for logs
    swing_pretty = swing.replace("->", "→").replace("VALLEY", "VALLEY").replace("PEAK", "PEAK")
    swing_pretty = swing_pretty.replace("VALLEY→PEAK", "VALLEY→PEAK").replace("PEAK→VALLEY", "PEAK→VALLEY")

    return {"prev": prev, "last": last, "swing": swing_pretty}
