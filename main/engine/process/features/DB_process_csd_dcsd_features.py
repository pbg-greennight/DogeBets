from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from main.engine.process.features.DB_process_v21_common import SIGMAS_ALL, SIGMAS_FAST, SIGMAS_SLOW
except Exception:  # pragma: no cover - local fallback for standalone testing
    from process.features.DB_process_v21_common import SIGMAS_ALL, SIGMAS_FAST, SIGMAS_SLOW


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _sign3(x: Any, eps: float = 1e-12) -> int:
    v = _safe_float(x, 0.0) or 0.0
    if v > eps:
        return 1
    if v < -eps:
        return -1
    return 0


def _mean_clean(values) -> Optional[float]:
    xs = []
    for value in values:
        fv = _safe_float(value)
        if fv is not None:
            xs.append(fv)
    return (sum(xs) / len(xs)) if xs else None


def _count_series_sign_flips(series) -> Optional[int]:
    if not isinstance(series, (list, tuple)) or len(series) < 2:
        return None
    cleaned = []
    for v in series:
        fv = _safe_float(v)
        if fv is None:
            continue
        if fv > 0:
            cleaned.append(1)
        elif fv < 0:
            cleaned.append(-1)
        else:
            cleaned.append(0)
    if len(cleaned) < 2:
        return None
    flips = 0
    prev = cleaned[0]
    for cur in cleaned[1:]:
        if cur != 0 and prev != 0 and cur != prev:
            flips += 1
        if cur != 0:
            prev = cur
    return flips


def _extract_leg_sigma_metrics(leg_payload: dict, sigma: int) -> dict:
    sigma_pack = (leg_payload or {}).get(int(sigma))
    if sigma_pack is None and isinstance(leg_payload, dict):
        sigma_pack = leg_payload.get(str(sigma))
    sigma_pack = sigma_pack or {}

    mid = [_safe_float(v) for v in (sigma_pack.get("mid") or []) if _safe_float(v) is not None]
    width = [_safe_float(v) for v in (sigma_pack.get("width") or []) if _safe_float(v) is not None]
    d_mid = [_safe_float(v) for v in (sigma_pack.get("d_mid") or []) if _safe_float(v) is not None]
    d_width = [_safe_float(v) for v in (sigma_pack.get("d_width") or []) if _safe_float(v) is not None]

    mid_delta = None
    if len(mid) >= 2:
        mid_delta = mid[-1] - mid[0]
    width_delta = None
    if len(width) >= 2:
        width_delta = width[-1] - width[0]

    width_first = width[0] if width else None
    width_last = width[-1] if width else None
    shrink_ratio = None
    if width_first is not None and abs(width_first) > 1e-9 and width_last is not None:
        shrink_ratio = width_last / width_first

    return {
        "mid_mean": _mean_clean(mid),
        "width_mean": _mean_clean(width),
        "mid_delta": mid_delta,
        "width_delta": width_delta,
        "width_last": width_last,
        "samples": len(mid) if mid else len(width),
        "dmid_mean": _mean_clean(d_mid[1:] if len(d_mid) > 1 else d_mid),
        "dwidth_mean": _mean_clean(d_width[1:] if len(d_width) > 1 else d_width),
        "dmid_last": d_mid[-1] if d_mid else None,
        "dwidth_last": d_width[-1] if d_width else None,
        "dmid_flip_count": _count_series_sign_flips(d_mid),
        "dwidth_flip_count": _count_series_sign_flips(d_width),
        "shrink_ratio": shrink_ratio,
    }


def _summarize_group(per_sigma: dict, sigmas: list[int], prefix: str) -> dict:
    def vals(key):
        return [per_sigma.get(s, {}).get(key) for s in sigmas]

    return {
        f"{prefix}_mid_mean": _mean_clean(vals("mid_mean")),
        f"{prefix}_width_mean": _mean_clean(vals("width_mean")),
        f"{prefix}_mid_delta": _mean_clean(vals("mid_delta")),
        f"{prefix}_width_delta": _mean_clean(vals("width_delta")),
        f"{prefix}_dmid_mean": _mean_clean(vals("dmid_mean")),
        f"{prefix}_dwidth_mean": _mean_clean(vals("dwidth_mean")),
        f"{prefix}_dmid_flip_density": _mean_clean(vals("dmid_flip_count")),
        f"{prefix}_dwidth_flip_density": _mean_clean(vals("dwidth_flip_count")),
    }


def _compute_release_score(leg1: dict, leg2: dict) -> float:
    flags = []
    for sigma in SIGMAS_FAST:
        l1 = leg1.get(sigma, {})
        l2 = leg2.get(sigma, {})
        cond = (
            (_safe_float(l2.get("width_delta"), 0.0) or 0.0) > 0.0
            and _sign3(l1.get("mid_delta")) != 0
            and _sign3(l2.get("mid_delta")) == _sign3(l1.get("mid_delta"))
        )
        flags.append(1.0 if cond else 0.0)
    return (sum(flags) / len(flags)) if flags else 0.0


def _compute_reversal_stage(leg1: dict, leg2: dict) -> float:
    flags = []
    for sigma in SIGMAS_FAST:
        l1 = leg1.get(sigma, {})
        l2 = leg2.get(sigma, {})
        cond = (
            _sign3(l1.get("mid_delta")) != 0
            and _sign3(l2.get("mid_delta")) == -_sign3(l1.get("mid_delta"))
            and (_safe_float(l2.get("width_delta"), 0.0) or 0.0) >= 0.0
        )
        flags.append(1.0 if cond else 0.0)
    return (sum(flags) / len(flags)) if flags else 0.0


def _compute_same_sign_exhaustion(leg1: dict, leg2: dict) -> float:
    flags = []
    for sigma in SIGMAS_FAST:
        l1 = leg1.get(sigma, {})
        l2 = leg2.get(sigma, {})
        l1_mid = _safe_float(l1.get("mid_delta"), 0.0) or 0.0
        l2_mid = _safe_float(l2.get("mid_delta"), 0.0) or 0.0
        l2_wd = _safe_float(l2.get("width_delta"), 0.0) or 0.0
        cond = (
            _sign3(l1_mid) != 0
            and _sign3(l2_mid) == _sign3(l1_mid)
            and abs(l2_mid) < abs(l1_mid)
            and l2_wd < 0.0
        )
        flags.append(1.0 if cond else 0.0)
    return (sum(flags) / len(flags)) if flags else 0.0


def build_csd_dcsd_feature_payload(
    pv_tail: dict,
    config: Optional[dict] = None,
) -> dict:
    leg1_raw = (pv_tail or {}).get("per_sigma_leg1") or {}
    leg2_raw = (pv_tail or {}).get("per_sigma_leg2") or (pv_tail or {}).get("per_sigma") or {}

    leg1: Dict[int, Dict[str, Any]] = {}
    leg2: Dict[int, Dict[str, Any]] = {}

    for sigma in SIGMAS_ALL:
        leg1[sigma] = _extract_leg_sigma_metrics(leg1_raw, sigma)
        leg2[sigma] = _extract_leg_sigma_metrics(leg2_raw, sigma)

    release_score = _compute_release_score(leg1, leg2)
    reversal_stage = _compute_reversal_stage(leg1, leg2)
    same_sign_exhaustion = _compute_same_sign_exhaustion(leg1, leg2)

    spread_now = _mean_clean([leg2.get(s, {}).get("width_last") for s in SIGMAS_FAST])
    spread_delta = _mean_clean([leg2.get(s, {}).get("width_delta") for s in SIGMAS_FAST])

    return {
        "leg1": leg1,
        "leg2": leg2,
        "summary": {
            "leg1_fast": _summarize_group(leg1, SIGMAS_FAST, "l1_fast"),
            "leg1_slow": _summarize_group(leg1, SIGMAS_SLOW, "l1_slow"),
            "leg2_fast": _summarize_group(leg2, SIGMAS_FAST, "l2_fast"),
            "leg2_slow": _summarize_group(leg2, SIGMAS_SLOW, "l2_slow"),
            "spread_now": spread_now,
            "spread_delta": spread_delta,
            "release_score": release_score,
            "reversal_stage": reversal_stage,
            "same_sign_exhaustion": same_sign_exhaustion,
        },
    }


def flatten_csd_dcsd_to_src(
    csd_obj: dict,
    config: Optional[dict] = None,
) -> dict:
    out: Dict[str, Any] = {}

    for leg_name in ["leg1", "leg2"]:
        leg = csd_obj.get(leg_name, {})
        leg_prefix = "l1" if leg_name == "leg1" else "l2"
        for sigma, fields in leg.items():
            out[f"src_csd_{leg_prefix}_mid_mean_s{sigma}"] = fields.get("mid_mean")
            out[f"src_csd_{leg_prefix}_width_mean_s{sigma}"] = fields.get("width_mean")
            out[f"src_csd_{leg_prefix}_width_last_s{sigma}"] = fields.get("width_last")
            out[f"src_csd_{leg_prefix}_width_delta_s{sigma}"] = fields.get("width_delta")
            out[f"src_csd_{leg_prefix}_mid_delta_s{sigma}"] = fields.get("mid_delta")
            out[f"src_csd_{leg_prefix}_samples_s{sigma}"] = fields.get("samples")
            out[f"src_dcsd_{leg_prefix}_dmid_mean_s{sigma}"] = fields.get("dmid_mean")
            out[f"src_dcsd_{leg_prefix}_dwidth_mean_s{sigma}"] = fields.get("dwidth_mean")
            out[f"src_dcsd_{leg_prefix}_dmid_last_s{sigma}"] = fields.get("dmid_last")
            out[f"src_dcsd_{leg_prefix}_dwidth_last_s{sigma}"] = fields.get("dwidth_last")
            out[f"src_dcsd_{leg_prefix}_dmid_flip_count_s{sigma}"] = fields.get("dmid_flip_count")
            out[f"src_dcsd_{leg_prefix}_dwidth_flip_count_s{sigma}"] = fields.get("dwidth_flip_count")
            if leg_prefix == "l2":
                out[f"src_csd_shrink_s{sigma}"] = fields.get("shrink_ratio")

    summary = csd_obj.get("summary", {})
    out["src_csd_spread_now"] = summary.get("spread_now")
    out["src_csd_spread_delta"] = summary.get("spread_delta")
    out["src_dcsd_release_score"] = summary.get("release_score")
    out["src_dcsd_reversal_stage"] = summary.get("reversal_stage")
    out["src_dcsd_same_sign_exhaustion"] = summary.get("same_sign_exhaustion")

    l2_fast = summary.get("leg2_fast", {})
    l2_slow = summary.get("leg2_slow", {})
    out["src_csd_l2_fast_width_mean"] = l2_fast.get("l2_fast_width_mean")
    out["src_csd_l2_slow_width_mean"] = l2_slow.get("l2_slow_width_mean")

    # Preserve prior summary flattening pattern for compatibility with exporters.
    for section_name, section in summary.items():
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            out[f"src_csd_{section_name}_{key}"] = value

    return out
