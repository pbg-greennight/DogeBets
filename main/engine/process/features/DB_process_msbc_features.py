from __future__ import annotations

from typing import Any, Dict, List, Optional

from main.engine.process.features.DB_process_v21_common import SIGMAS_ALL, SIGMAS_FAST, SIGMAS_SLOW


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


def _safe_mean(values: List[Optional[float]], default: Optional[float] = None) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return default
    return float(sum(xs) / len(xs))


def _mono(values: List[Optional[float]]) -> float:
    xs = [float(v) for v in values if v is not None]
    if len(xs) < 2:
        return 0.0
    diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    if not diffs:
        return 0.0
    ok = sum(1 for d in diffs if d >= -_EPS)
    return float(ok) / float(len(diffs))


def _label_age(age_seconds: Optional[float]) -> Optional[str]:
    if age_seconds is None:
        return None
    if age_seconds < 300:
        return "fresh"
    if age_seconds < 1800:
        return "active"
    if age_seconds < 7200:
        return "mature"
    return "stale"


def _pack_for_sigma(root: Any, sigma: int) -> dict:
    if isinstance(root, dict):
        return (root.get(int(sigma)) or root.get(str(sigma)) or {}) if root else {}
    return {}


def _extract_sigma_leg_metrics(bell: dict, sigma: int) -> dict:
    leg1_root = (((bell or {}).get("leg1") or {}).get("sigmas") or {})
    leg2_root = (((bell or {}).get("leg2") or {}).get("sigmas") or {})

    leg1_pack = _pack_for_sigma(leg1_root, sigma)
    leg2_pack = _pack_for_sigma(leg2_root, sigma)
    m1 = (leg1_pack.get("metrics") or {}) if isinstance(leg1_pack, dict) else {}
    m2 = (leg2_pack.get("metrics") or {}) if isinstance(leg2_pack, dict) else {}

    l1_slope = _f(m1.get("slope"))
    l2_slope = _f(m2.get("slope"))
    l1_curve = _f(m1.get("curve"))
    l2_curve = _f(m2.get("curve"))
    l1_tan = _f(m1.get("quad_tangent"))
    l2_tan = _f(m2.get("quad_tangent"))
    l1_curv = _f(m1.get("quad_curv"))
    l2_curv = _f(m2.get("quad_curv"))

    sign_agree = int(_sgn(l1_slope) != 0 and _sgn(l1_slope) == _sgn(l2_slope))
    accel_agree = int(_sgn(l1_curve) != 0 and _sgn(l1_curve) == _sgn(l2_curve))
    tan_agree = int(_sgn(l1_tan) != 0 and _sgn(l1_tan) == _sgn(l2_tan))
    override = int(_sgn(l1_slope) != 0 and _sgn(l2_slope) != 0 and _sgn(l1_slope) != _sgn(l2_slope))

    slope_ratio = None
    if l1_slope is not None and abs(l1_slope) > _EPS and l2_slope is not None:
        slope_ratio = float(l2_slope / l1_slope)

    continuity_bits: List[Optional[float]] = [
        float(sign_agree),
        float(accel_agree),
        float(tan_agree),
        _f(m1.get("lin_r2")),
        _f(m2.get("lin_r2")),
    ]
    continuity = _safe_mean(continuity_bits, default=None)

    return {
        "l1_slope": l1_slope,
        "l2_slope": l2_slope,
        "l1_curve": l1_curve,
        "l2_curve": l2_curve,
        "l1_lin_r2": _f(m1.get("lin_r2")),
        "l2_lin_r2": _f(m2.get("lin_r2")),
        "l1_quad_tangent": l1_tan,
        "l2_quad_tangent": l2_tan,
        "l1_quad_curv": l1_curv,
        "l2_quad_curv": l2_curv,
        "l1_z": _f(m1.get("z_last")),
        "l2_z": _f(m2.get("z_last")),
        "l1_run_score": _f(m1.get("run_score")),
        "l2_run_score": _f(m2.get("run_score")),
        "l2_tag": m2.get("tag"),
        "override": override,
        "sign_agree": sign_agree,
        "accel_agree": accel_agree,
        "slope_ratio": slope_ratio,
        "continuity": continuity,
    }


def _compute_transfer_summary(bell: dict) -> dict:
    leg2_root = (((bell or {}).get("leg2") or {}).get("sigmas") or {})
    sigmas = [s for s in SIGMAS_ALL if _pack_for_sigma(leg2_root, s)]
    if not sigmas:
        return {
            "transfer_dir": "none",
            "transfer_depth": 0.0,
            "transfer_state": "none",
        }

    metrics_map = {s: ((_pack_for_sigma(leg2_root, s).get("metrics") or {}) if _pack_for_sigma(leg2_root, s) else {}) for s in sigmas}
    first = metrics_map.get(sigmas[0], {}) or {}
    dir_sign = _sgn(first.get("slope")) or _sgn(first.get("quad_tangent"))
    dir_label = "up" if dir_sign > 0 else ("down" if dir_sign < 0 else "none")

    depth_score = 0.0
    for sigma in sigmas[1:]:
        m = metrics_map.get(sigma, {}) or {}
        ok = 1 if dir_sign != 0 and (_sgn(m.get("slope")) == dir_sign or _sgn(m.get("quad_tangent")) == dir_sign) else 0
        depth_score += float(ok)

    if depth_score <= 0:
        state = "none"
    elif depth_score <= 1:
        state = "shallow"
    elif depth_score <= 2:
        state = "partial"
    elif depth_score <= 3:
        state = "deep"
    else:
        state = "full"

    return {
        "transfer_dir": dir_label,
        "transfer_depth": float(depth_score),
        "transfer_state": state,
    }


def _compute_propagation_summary(bell: dict) -> dict:
    leg2_root = (((bell or {}).get("leg2") or {}).get("sigmas") or {})
    sigmas = [s for s in SIGMAS_ALL if _pack_for_sigma(leg2_root, s)]
    metrics_map = {s: ((_pack_for_sigma(leg2_root, s).get("metrics") or {}) if _pack_for_sigma(leg2_root, s) else {}) for s in sigmas}

    slope_vals: List[Optional[float]] = []
    tan_vals: List[Optional[float]] = []
    curve_vals: List[Optional[float]] = []
    linm_vals: List[Optional[float]] = []

    pair_rows = 0
    agree_flags: List[int] = []
    disorder = 0

    for s in sigmas:
        m = metrics_map.get(s, {}) or {}
        if not m:
            continue
        slope_vals.append(_f(m.get("slope")))
        tan_vals.append(_f(m.get("quad_tangent")))
        curve_vals.append(_f(m.get("curve")))
        linm_vals.append(_f(m.get("lin_slope")))

    for i in range(len(sigmas) - 1):
        a = sigmas[i]
        b = sigmas[i + 1]
        ma = metrics_map.get(a, {}) or {}
        mb = metrics_map.get(b, {}) or {}
        if not ma or not mb:
            continue
        pair_rows += 1
        sa = int(_sgn(ma.get("slope")) == _sgn(mb.get("slope")) != 0)
        ta = int(_sgn(ma.get("quad_tangent")) == _sgn(mb.get("quad_tangent")) != 0)
        ca = int(_sgn(ma.get("curve")) == _sgn(mb.get("curve")) != 0)
        la = int(_sgn(ma.get("lin_slope")) == _sgn(mb.get("lin_slope")) != 0)
        agree_flags.extend([sa, ta, ca, la])
        disorder += (1 - sa) + (1 - ta) + (1 - ca) + (1 - la)

    prop_agree = float(sum(agree_flags) / len(agree_flags)) if agree_flags else 0.0
    prop_disagree = 1.0 - prop_agree if agree_flags else 0.0

    return {
        "propagation_agree_ratio": prop_agree,
        "propagation_disagree_ratio": prop_disagree,
        "pair_rows": pair_rows,
        "slope_mono": _mono(slope_vals),
        "tan_mono": _mono(tan_vals),
        "curve_decay": _mono([(-v if v is not None else None) for v in curve_vals]),
        "linm_mono": _mono(linm_vals),
        "disorder_count": disorder,
    }


def _compute_consistency_summary(per_sigma: dict) -> dict:
    overrides = {s: per_sigma.get(s, {}).get("override") for s in SIGMAS_ALL}
    continuity_vals = [
        _f((per_sigma.get(s, {}) or {}).get("continuity"))
        for s in SIGMAS_ALL
        if per_sigma.get(s)
    ]
    return {
        "override_count": sum(1 for v in overrides.values() if bool(v)),
        "fast_override_count": sum(1 for s in SIGMAS_FAST if bool(overrides.get(s))),
        "slow_override_count": sum(1 for s in SIGMAS_SLOW if bool(overrides.get(s))),
        "overrides": overrides,
        "continuity_mean": _safe_mean(continuity_vals, default=0.0),
    }


def _compute_age_summary(bell_curve_series: Optional[dict]) -> dict:
    spans: List[float] = []
    for sigma in SIGMAS_ALL:
        pack = ((bell_curve_series or {}).get(int(sigma)) or (bell_curve_series or {}).get(str(sigma)) or {})
        ts = (pack.get("ts") or []) if isinstance(pack, dict) else []
        if len(ts) >= 2:
            try:
                spans.append(float((ts[-1] - ts[0]).total_seconds()))
            except Exception:
                pass
    age_seconds = max(spans) if spans else None
    return {
        "age_seconds": age_seconds,
        "age_label": _label_age(age_seconds),
    }


def build_msbc_feature_payload(
    bell: dict,
    bell_curve_series: Optional[dict] = None,
    config: Optional[dict] = None,
) -> dict:
    per_sigma: Dict[int, Dict[str, Any]] = {}
    for sigma in SIGMAS_ALL:
        per_sigma[sigma] = _extract_sigma_leg_metrics(bell, sigma)

    transfer = _compute_transfer_summary(bell)
    propagation = _compute_propagation_summary(bell)
    consistency = _compute_consistency_summary(per_sigma)
    age = _compute_age_summary(bell_curve_series)

    payload = {
        "per_sigma": per_sigma,
        "transfer": transfer,
        "propagation": propagation,
        "consistency": consistency,
        "age": age,
        "meta": {
            "disorder_count": int(propagation.get("disorder_count") or 0),
        },
    }
    return payload


def flatten_msbc_to_src(
    msbc_obj: dict,
    config: Optional[dict] = None,
) -> dict:
    out: Dict[str, Any] = {}

    per_sigma = msbc_obj.get("per_sigma", {})
    for sigma, fields in per_sigma.items():
        out[f"src_msbc_l1_slope_s{sigma}"] = fields.get("l1_slope")
        out[f"src_msbc_l2_slope_s{sigma}"] = fields.get("l2_slope")
        out[f"src_msbc_l1_curve_s{sigma}"] = fields.get("l1_curve")
        out[f"src_msbc_l2_curve_s{sigma}"] = fields.get("l2_curve")
        out[f"src_msbc_l1_lin_r2_s{sigma}"] = fields.get("l1_lin_r2")
        out[f"src_msbc_l2_lin_r2_s{sigma}"] = fields.get("l2_lin_r2")
        out[f"src_msbc_l1_quad_tangent_s{sigma}"] = fields.get("l1_quad_tangent")
        out[f"src_msbc_l2_quad_tangent_s{sigma}"] = fields.get("l2_quad_tangent")
        out[f"src_msbc_l1_quad_curv_s{sigma}"] = fields.get("l1_quad_curv")
        out[f"src_msbc_l2_quad_curv_s{sigma}"] = fields.get("l2_quad_curv")
        out[f"src_msbc_l1_z_s{sigma}"] = fields.get("l1_z")
        out[f"src_msbc_l2_z_s{sigma}"] = fields.get("l2_z")
        out[f"src_msbc_l1_run_score_s{sigma}"] = fields.get("l1_run_score")
        out[f"src_msbc_l2_run_score_s{sigma}"] = fields.get("l2_run_score")
        out[f"src_msbc_l2_tag_s{sigma}"] = fields.get("l2_tag")
        out[f"src_msbc_override_s{sigma}"] = fields.get("override")
        out[f"src_msbc_sign_agree_s{sigma}"] = fields.get("sign_agree")
        out[f"src_msbc_accel_agree_s{sigma}"] = fields.get("accel_agree")
        out[f"src_msbc_slope_ratio_s{sigma}"] = fields.get("slope_ratio")
        out[f"src_msbc_continuity_s{sigma}"] = fields.get("continuity")

    transfer = msbc_obj.get("transfer", {})
    out["src_msbc_transfer_dir"] = transfer.get("transfer_dir")
    out["src_msbc_transfer_depth"] = transfer.get("transfer_depth")
    out["src_msbc_transfer_state"] = transfer.get("transfer_state")

    propagation = msbc_obj.get("propagation", {})
    out["src_msbc_propagation_agree_ratio"] = propagation.get("propagation_agree_ratio")
    out["src_msbc_propagation_disagree_ratio"] = propagation.get("propagation_disagree_ratio")
    out["src_msbc_pair_rows"] = propagation.get("pair_rows")
    out["src_msbc_slope_mono"] = propagation.get("slope_mono")
    out["src_msbc_tan_mono"] = propagation.get("tan_mono")
    out["src_msbc_curve_decay"] = propagation.get("curve_decay")
    out["src_msbc_linm_mono"] = propagation.get("linm_mono")

    consistency = msbc_obj.get("consistency", {})
    out["src_msbc_override_count"] = consistency.get("override_count")
    out["src_msbc_fast_override_count"] = consistency.get("fast_override_count")
    out["src_msbc_slow_override_count"] = consistency.get("slow_override_count")
    out["src_msbc_continuity_mean"] = consistency.get("continuity_mean")

    meta = msbc_obj.get("meta", {})
    out["src_msbc_disorder_count"] = meta.get("disorder_count")

    age = msbc_obj.get("age", {})
    out["src_msbc_age_seconds"] = age.get("age_seconds")
    out["src_msbc_age_label"] = age.get("age_label")

    return out
