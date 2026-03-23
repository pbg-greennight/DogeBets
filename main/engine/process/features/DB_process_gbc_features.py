from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from main.engine.process.features.DB_process_v21_common import SIGMAS_ALL


_EPS = 1e-12


def _f(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if v == v else None
    except Exception:
        return None


def _sgn(x: Any, eps: float = _EPS) -> int:
    v = _f(x)
    if v is None:
        return 0
    if v > eps:
        return 1
    if v < -eps:
        return -1
    return 0


def _pack_for_sigma(root: Any, sigma: int) -> dict:
    if isinstance(root, dict):
        return (root.get(int(sigma)) or root.get(str(sigma)) or {}) if root else {}
    return {}


def _seg_slope(vals: List[float], ts: List[datetime], k: int) -> float:
    if len(vals) < 2:
        return 0.0
    kk = max(1, min(int(k), len(vals) - 1))
    if ts and len(ts) >= len(vals):
        try:
            dt = max((ts[-1] - ts[-1 - kk]).total_seconds(), 1e-9)
        except Exception:
            dt = float(kk)
    else:
        dt = float(kk)
    return (float(vals[-1]) - float(vals[-1 - kk])) / dt


def _tail_diagnostics_from_series(ts: List[Any], vals: List[Any]) -> dict:
    raw_vals = [float(v) for v in (vals or []) if v is not None]
    raw_ts = list(ts or [])[-len(raw_vals):]
    if len(raw_vals) < 3:
        return {
            "shrink": 0.0,
            "flat": 0.0,
            "hook": 0,
            "prev_abs": 0.0,
            "last_abs": 0.0,
            "sign_from": 0,
            "sign_to": 0,
            "eps": 0.0,
        }

    tail_vals = list(raw_vals[-max(20, min(120, len(raw_vals))):])
    tail_ts = list(raw_ts[-len(tail_vals):]) if raw_ts else []
    k_seg = max(6, min(20, len(tail_vals) // 6))

    prev_slice_vals = tail_vals[:-k_seg] if len(tail_vals) > 2 * k_seg else tail_vals
    prev_slice_ts = tail_ts[:-k_seg] if tail_ts and len(tail_vals) > 2 * k_seg else tail_ts

    prev_slope = _seg_slope(prev_slice_vals, prev_slice_ts, k=k_seg)
    last_slope = _seg_slope(tail_vals, tail_ts, k=k_seg)

    prev_abs = abs(prev_slope)
    last_abs = abs(last_slope)
    shrink = (last_abs / (prev_abs + 1e-9)) if prev_abs > 0 else 0.0

    sign_from = 0 if prev_slope == 0 else (1 if prev_slope > 0 else -1)
    sign_to = 0 if last_slope == 0 else (1 if last_slope > 0 else -1)

    short_s = _seg_slope(tail_vals, tail_ts, k=max(4, k_seg // 2))
    long_s = _seg_slope(tail_vals, tail_ts, k=max(12, k_seg * 2))
    hook = 1 if (short_s != 0.0 and long_s != 0.0 and (short_s > 0) != (long_s > 0)) else 0

    diffs: List[float] = []
    for i in range(1, len(tail_vals)):
        if tail_ts and i < len(tail_ts):
            try:
                dt_i = max((tail_ts[i] - tail_ts[i - 1]).total_seconds(), 1e-9)
            except Exception:
                dt_i = 1.0
        else:
            dt_i = 1.0
        diffs.append((float(tail_vals[i]) - float(tail_vals[i - 1])) / dt_i)

    eps = 0.0
    flat = 0.0
    if len(diffs) >= 3:
        import statistics

        med = statistics.median(diffs)
        mad = statistics.median([abs(d - med) for d in diffs]) + 1e-9
        eps = float(mad)
        flat = max(0.0, min(1.0, 1.0 - (abs(last_slope) / (5.0 * mad))))

    return {
        "shrink": float(shrink),
        "flat": float(flat),
        "hook": int(hook),
        "prev_abs": float(prev_abs),
        "last_abs": float(last_abs),
        "sign_from": int(sign_from),
        "sign_to": int(sign_to),
        "eps": float(eps),
    }


def _sign_persistence(vals: List[Any]) -> int:
    xs = [float(v) for v in (vals or []) if v is not None]
    if len(xs) < 2:
        return 0
    diffs = [_sgn(xs[i] - xs[i - 1]) for i in range(1, len(xs))]
    diffs = [d for d in diffs if d != 0]
    if not diffs:
        return 0
    now = diffs[-1]
    age = 1
    for i in range(len(diffs) - 2, -1, -1):
        if diffs[i] != now:
            break
        age += 1
    return int(age)


def _bars_since_turn(vals: List[Any]) -> int:
    return _sign_persistence(vals)


def _hook_state(vals: List[Any], end: Optional[int] = None) -> int:
    xs = [float(v) for v in (vals or []) if v is not None]
    if end is not None:
        xs = xs[:end]
    if len(xs) < 12:
        return 0
    short_k = min(max(4, len(xs) // 8), max(1, len(xs) - 1))
    long_k = min(max(12, len(xs) // 3), max(1, len(xs) - 1))
    short_s = float(xs[-1] - xs[-1 - short_k]) / float(short_k)
    long_s = float(xs[-1] - xs[-1 - long_k]) / float(long_k)
    return int(short_s != 0.0 and long_s != 0.0 and ((short_s > 0) != (long_s > 0)))


def _hook_persistence(vals: List[Any]) -> int:
    xs = [float(v) for v in (vals or []) if v is not None]
    if len(xs) < 12:
        return 0
    current = _hook_state(xs)
    age = 1 if current else 0
    for end in range(len(xs) - 1, 11, -1):
        state = _hook_state(xs, end=end)
        if state != current:
            break
        if current:
            age += 1
    return int(age)


def _extract_diag_for_sigma(bell: dict, bell_curve_series: Optional[dict], sigma: int) -> dict:
    diag_root = (((bell or {}).get("diagnostics") or {}).get("per_sigma") or {})
    diag = _pack_for_sigma(diag_root, sigma)
    series_pack = _pack_for_sigma(bell_curve_series or {}, sigma)
    ts = (series_pack.get("ts") or []) if isinstance(series_pack, dict) else []
    vals = (series_pack.get("values") or []) if isinstance(series_pack, dict) else []

    derived = _tail_diagnostics_from_series(ts, vals)
    shrink = _f(diag.get("shrink"))
    flat = _f(diag.get("flat"))
    if shrink is None:
        shrink = derived.get("shrink")
    if flat is None:
        flat = derived.get("flat")

    norm_ctx = None
    if shrink is not None and flat is not None:
        norm_ctx = abs(float(shrink)) / (abs(float(flat)) + 1e-9)

    return {
        "shrink": shrink if shrink is not None else derived.get("shrink"),
        "flat": flat if flat is not None else derived.get("flat"),
        "hook": int(diag.get("hook")) if diag.get("hook") is not None else int(derived.get("hook") or 0),
        "prev_abs": _f(diag.get("prev_abs")) if _f(diag.get("prev_abs")) is not None else derived.get("prev_abs"),
        "last_abs": _f(diag.get("last_abs")) if _f(diag.get("last_abs")) is not None else derived.get("last_abs"),
        "sign_from": int(diag.get("sign_from")) if diag.get("sign_from") is not None else int(derived.get("sign_from") or 0),
        "sign_to": int(diag.get("sign_to")) if diag.get("sign_to") is not None else int(derived.get("sign_to") or 0),
        "eps": _f(diag.get("eps")) if _f(diag.get("eps")) is not None else derived.get("eps"),
        "turn_age": int(diag.get("turn_age")) if diag.get("turn_age") is not None else _bars_since_turn(vals),
        "hook_age": int(diag.get("hook_age")) if diag.get("hook_age") is not None else _hook_persistence(vals),
        "sign_persist": int(diag.get("sign_persist")) if diag.get("sign_persist") is not None else _sign_persistence(vals),
        "hook_persist": int(diag.get("hook_persist")) if diag.get("hook_persist") is not None else _hook_persistence(vals),
        "norm_ctx": _f(diag.get("norm_ctx")) if _f(diag.get("norm_ctx")) is not None else norm_ctx,
    }


def build_gbc_feature_payload(
    bell: dict,
    bell_curve_series: Optional[dict] = None,
    config: Optional[dict] = None,
) -> dict:
    per_sigma: Dict[int, Dict[str, Any]] = {}
    active_sigma_count = 0
    for sigma in SIGMAS_ALL:
        per_sigma[sigma] = _extract_diag_for_sigma(bell, bell_curve_series, sigma)
        fields = per_sigma[sigma]
        if any(v not in (None, 0, 0.0, "", False) for v in fields.values()):
            active_sigma_count += 1

    return {
        "per_sigma": per_sigma,
        "meta": {
            "series_available": bell_curve_series is not None,
            "active_sigma_count": active_sigma_count,
        },
    }


def flatten_gbc_to_src(
    gbc_obj: dict,
    config: Optional[dict] = None,
) -> dict:
    out: Dict[str, Any] = {}
    per_sigma = gbc_obj.get("per_sigma", {})
    for sigma, fields in per_sigma.items():
        out[f"src_gbc_shrink_s{sigma}"] = fields.get("shrink")
        out[f"src_gbc_flat_s{sigma}"] = fields.get("flat")
        out[f"src_gbc_hook_s{sigma}"] = fields.get("hook")
        out[f"src_gbc_prev_abs_s{sigma}"] = fields.get("prev_abs")
        out[f"src_gbc_last_abs_s{sigma}"] = fields.get("last_abs")
        out[f"src_gbc_sign_from_s{sigma}"] = fields.get("sign_from")
        out[f"src_gbc_sign_to_s{sigma}"] = fields.get("sign_to")
        out[f"src_gbc_eps_s{sigma}"] = fields.get("eps")
        out[f"src_gbc_turn_age_s{sigma}"] = fields.get("turn_age")
        out[f"src_gbc_hook_age_s{sigma}"] = fields.get("hook_age")
        out[f"src_gbc_sign_persist_s{sigma}"] = fields.get("sign_persist")
        out[f"src_gbc_hook_persist_s{sigma}"] = fields.get("hook_persist")
        out[f"src_gbc_norm_ctx_s{sigma}"] = fields.get("norm_ctx")
        # historical aliases kept for mapper/back-compat
        out[f"src_gbc_s{sigma}_shrink"] = fields.get("shrink")
        out[f"src_gbc_s{sigma}_hook"] = fields.get("hook")

    meta = gbc_obj.get("meta", {}) or {}
    out["src_gbc_series_available"] = 1.0 if meta.get("series_available") else 0.0
    out["src_gbc_active_sigma_count"] = int(meta.get("active_sigma_count") or 0)
    return out
