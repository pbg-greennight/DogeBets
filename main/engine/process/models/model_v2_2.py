from __future__ import annotations

from typing import Any, Dict, Optional


def _sign_label(v: float) -> str:
    if v > 0:
        return "Bull"
    if v < 0:
        return "Bear"
    return "Neutral"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_nested_float(payload: Dict[str, Any], path: str, default: Optional[float] = None) -> Optional[float]:
    cur: Any = payload
    for key in path.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur.get(key)
    try:
        return float(cur)
    except Exception:
        return default


def _base_v2_1_decision(features: Dict[str, Any], t: Dict[str, Any]) -> Dict[str, Any]:
    gauss = (features.get("gauss") or {})
    fan = (features.get("fan") or {})
    spacing = (features.get("spacing") or {})
    compression = (features.get("compression") or {})
    hysteresis = (features.get("hysteresis") or {})
    torque = (features.get("torque") or {})

    slopes = (gauss.get("slopes") or {})
    s23 = _f(slopes.get("s23"), 0.0)
    s53 = _f(slopes.get("s53"), 0.0)
    slope_sum = s23 + s53
    slope_magnitude = abs(slope_sum)

    fan_width = _f(fan.get("width_abs", fan.get("width", 0.0)), 0.0)
    alignment = _f(torque.get("alignment"), 0.0)
    comp_vel = _f(compression.get("velocity"), 0.0)
    flip_score = _f(hysteresis.get("flip_score"), 0.0)
    ratio_8_23_to_23_53 = _f(spacing.get("ratio_8_23_to_23_53"), 0.0)

    run_fan_threshold = _f(t.get("run_fan_threshold", 0.01), 0.01)
    noise_fan_threshold = _f(t.get("noise_fan_threshold", 0.004), 0.004)
    compression_threshold = _f(t.get("compression_threshold", 0.003), 0.003)
    flip_score_threshold = _f(t.get("flip_score_threshold", 0.5), 0.5)
    slope_noise_threshold = _f(t.get("slope_noise_threshold", 0.002), 0.002)
    alignment_threshold = _f(t.get("alignment_threshold", 0.5), 0.5)

    bull_run_flip_score_threshold = _f(t.get("bull_run_flip_score_threshold", 0.18), 0.18)
    bull_run_compression_min = _f(t.get("bull_run_compression_min", 0.0), 0.0)
    bull_run_ratio_max = _f(t.get("bull_run_ratio_max", -0.10), -0.10)

    regime = "NOISE"
    if fan_width > run_fan_threshold and alignment >= alignment_threshold:
        regime = "RUN"
    elif comp_vel > compression_threshold and flip_score >= flip_score_threshold:
        regime = "REVERSAL"
    elif fan_width < noise_fan_threshold and slope_magnitude <= slope_noise_threshold:
        regime = "NOISE"

    run_direction = _sign_label(slope_sum)
    if regime == "RUN":
        trend = run_direction
    elif regime == "REVERSAL":
        if run_direction == "Bull":
            trend = "Bear"
        elif run_direction == "Bear":
            trend = "Bull"
        else:
            trend = "Neutral"
    else:
        trend = "Neutral"

    bull_run_instability_override = bool(
        trend == "Bull"
        and regime == "RUN"
        and flip_score >= bull_run_flip_score_threshold
        and comp_vel >= bull_run_compression_min
        and ratio_8_23_to_23_53 <= bull_run_ratio_max
    )
    if bull_run_instability_override:
        trend = "Bear"

    return {
        "trend": trend,
        "regime": regime,
        "run_direction": run_direction,
        "slope_sum": slope_sum,
        "slope_magnitude": slope_magnitude,
        "fan_width": fan_width,
        "alignment": alignment,
        "compression_velocity": comp_vel,
        "flip_score": flip_score,
        "ratio_8_23_to_23_53": ratio_8_23_to_23_53,
        "v2_1_override": bull_run_instability_override,
    }


def compute_reversal_pressure(fan_width_acceleration: Optional[float], hinge_velocity: Optional[float], snap_velocity: Optional[float]) -> float:
    valid: list[float] = []
    if fan_width_acceleration is not None:
        valid.append(-fan_width_acceleration)
    if hinge_velocity is not None:
        valid.append(abs(hinge_velocity))
    if snap_velocity is not None:
        valid.append(snap_velocity)
    if not valid:
        return 0.0
    return float(sum(valid))


def detect_false_snap(fx: Dict[str, float], t: Dict[str, float]) -> bool:
    return bool(
        fx["snap_score"] >= t["snap_score_high"]
        and fx["hinge_torsion"] <= t["false_snap_hinge_max"]
        and fx["slow_retention"] >= t["slow_retention_high"]
        and fx["outer_order_intact"] >= 1.0
    )


def detect_true_snap(fx: Dict[str, float], t: Dict[str, float]) -> bool:
    return bool(
        fx["snap_score"] >= t["snap_score_high"]
        and fx["snap_velocity"] >= t["snap_velocity_min"]
        and fx["hinge_torsion"] >= t["true_snap_hinge_min"]
        and abs(fx["hinge_velocity"]) >= t["hinge_velocity_min"]
        and fx["reversal_pressure"] >= t["reversal_pressure_high"]
        and fx["outer_resistance"] < 2.0
    )


def detect_inertia_wall(fx: Dict[str, float], t: Dict[str, float]) -> bool:
    return bool(
        fx["snap_score"] >= t["snap_score_high"]
        and fx["slow_retention"] >= t["slow_retention_high"]
        and fx["outer_resistance"] >= 2.0
        and fx["hinge_torsion"] <= t["hinge_torsion_high"]
    )


def build_v2_2_features(features: Dict[str, Any], baseline: Dict[str, Any], t: Dict[str, float]) -> Dict[str, float]:
    physics = (((features.get("history") or {}).get("fan_physics") or {}))
    geo = (physics.get("geometry") or {})
    hinge = (physics.get("hinge") or {})
    snap = (physics.get("snap") or {})
    phase = (physics.get("phase") or {})
    energy = (physics.get("energy") or {})

    gauss = features.get("gauss") or {}
    curv = (gauss.get("curvature") or {})

    fan_width_velocity = _extract_nested_float(features, "fan.width_velocity")
    fan_width_acceleration = _extract_nested_float(features, "fan.width_acceleration")
    if fan_width_velocity is None:
        fan_width_velocity = _f(geo.get("fan_width_velocity"), 0.0)
    if fan_width_acceleration is None:
        fan_width_acceleration = _f(geo.get("fan_width_acceleration"), 0.0)

    hinge_velocity = _extract_nested_float(features, "fan.hinge_velocity")
    if hinge_velocity is None:
        hinge_velocity = _f(hinge.get("hinge_velocity"), 0.0)

    snap_velocity = _extract_nested_float(features, "fan.snap_velocity")
    if snap_velocity is None:
        snap_velocity = _f(snap.get("snap_velocity"), 0.0)

    outer_order_signature = str(geo.get("outer_order_signature") or "")
    outer_order_intact = 1.0 if ("83>68>53" in outer_order_signature or "53>68>83" in outer_order_signature) else 0.0

    outer_resistance = 0.0
    if _f(energy.get("slow_retention"), 0.0) >= t["slow_retention_high"]:
        outer_resistance += 1.0
    if _f(geo.get("outer_fan_width"), 0.0) >= t["outer_fan_width_min"]:
        outer_resistance += 1.0
    if outer_order_intact >= 1.0:
        outer_resistance += 1.0

    reversal_pressure = compute_reversal_pressure(fan_width_acceleration, hinge_velocity, snap_velocity)

    return {
        "fan_width_velocity": _f(fan_width_velocity, 0.0),
        "fan_width_acceleration": _f(fan_width_acceleration, 0.0),
        "fan_order_score": _f(geo.get("fan_order_score"), 0.0),
        "outer_fan_width": _f(geo.get("outer_fan_width"), 0.0),
        "hinge_velocity": _f(hinge_velocity, 0.0),
        "hinge_conflict": 1.0 if bool(hinge.get("hinge_conflict")) else 0.0,
        "hinge_torsion": _f(hinge.get("hinge_torsion"), 0.0),
        "snap_score": _f(snap.get("snap_score"), 0.0),
        "snap_velocity": _f(snap_velocity, 0.0),
        "phase_alignment": _f(phase.get("phase_alignment"), baseline["alignment"]),
        "fan_phase_score": _f(phase.get("fan_phase_score"), 0.0),
        "phase_disagreement": 1.0 if bool(phase.get("phase_disagreement")) else 0.0,
        "slow_retention": _f(energy.get("slow_retention"), 0.0),
        "g83_curvature": _f(curv.get("s83"), 0.0),
        "outer_order_signature": outer_order_signature,
        "outer_order_intact": outer_order_intact,
        "outer_resistance": outer_resistance,
        "reversal_pressure": reversal_pressure,
    }


def apply_v2_2_rules(base_trend: str, baseline: Dict[str, Any], fx: Dict[str, float], t: Dict[str, float]) -> Dict[str, Any]:
    trend = base_trend
    rule = "BASELINE_V2_1"
    counts = {
        "continuation_protection_blocks": 0,
        "collapse_reversal_overrides": 0,
        "false_snap_blocks": 0,
        "inertia_wall_neutrals": 0,
        "reversal_pressure_overrides": 0,
        "neutral_promotions": 0,
    }

    continuation_ok = (
        baseline["regime"] == "RUN"
        and fx["hinge_conflict"] < 0.5
        and fx["hinge_torsion"] < t["hinge_torsion_high"]
        and fx["reversal_pressure"] < t["reversal_pressure_high"]
        and fx["phase_alignment"] >= t["phase_alignment_min"]
        and fx["fan_phase_score"] >= t["fan_phase_score_min"]
    )
    if continuation_ok:
        if baseline["run_direction"] == "Bull" and fx["fan_width_velocity"] >= t["fan_width_velocity_expand"] and base_trend == "Bear":
            trend = "Neutral"
            rule = "RUN_CONTINUATION_PROTECTION"
            counts["continuation_protection_blocks"] += 1
        if baseline["run_direction"] == "Bear" and fx["fan_width_velocity"] <= -t["fan_width_velocity_expand"] and base_trend == "Bull":
            trend = "Neutral"
            rule = "RUN_CONTINUATION_PROTECTION"
            counts["continuation_protection_blocks"] += 1

    collapse_candidate = (
        trend == "Bull"
        and fx["fan_width_velocity"] <= t["fan_width_velocity_collapse"]
        and fx["fan_width_acceleration"] <= t["fan_width_acceleration_collapse"]
        and fx["snap_score"] >= t["snap_score_high"]
        and fx["hinge_torsion"] >= t["hinge_torsion_high"]
        and fx["reversal_pressure"] >= t["reversal_pressure_high"]
    )
    if collapse_candidate:
        trend = "Neutral" if fx["outer_resistance"] >= 2.0 else "Bear"
        rule = "COLLAPSE_REVERSAL"
        counts["collapse_reversal_overrides"] += 1

    false_snap = detect_false_snap(fx, t)
    true_snap = detect_true_snap(fx, t)
    if false_snap:
        if trend in ("Bear", "Bull"):
            trend = "Neutral"
        rule = "FALSE_SNAP_BLOCK"
        counts["false_snap_blocks"] += 1
    elif true_snap and baseline["run_direction"] == "Bull" and trend == "Bull":
        trend = "Bear"
        rule = "TRUE_SNAP_REVERSAL"

    inertia_wall = detect_inertia_wall(fx, t)
    if inertia_wall and trend in ("Bear", "Bull"):
        trend = "Neutral"
        rule = "INERTIA_WALL_NEUTRAL"
        counts["inertia_wall_neutrals"] += 1

    rp_override = (
        fx["reversal_pressure"] >= t["reversal_pressure_high"]
        and fx["snap_velocity"] >= t["snap_velocity_min"]
        and abs(fx["hinge_velocity"]) >= t["hinge_velocity_min"]
        and fx["fan_width_acceleration"] <= t["fan_width_acceleration_collapse"]
    )
    if rp_override and baseline["regime"] == "RUN":
        trend = "Neutral" if fx["outer_resistance"] >= 2.0 else ("Bear" if baseline["run_direction"] == "Bull" else "Bull")
        rule = "REVERSAL_PRESSURE_OVERRIDE"
        counts["reversal_pressure_overrides"] += 1

    if (
        trend == "Neutral"
        and baseline["regime"] == "RUN"
        and abs(baseline["slope_sum"]) >= t["neutral_promotion_min_strength"]
        and fx["phase_alignment"] >= t["phase_alignment_min"]
        and fx["fan_phase_score"] >= t["fan_phase_score_min"]
        and fx["reversal_pressure"] < t["reversal_pressure_high"]
        and fx["hinge_conflict"] < 0.5
    ):
        trend = baseline["run_direction"] if baseline["run_direction"] in ("Bull", "Bear") else "Neutral"
        rule = "STRONG_RUN_PROMOTION"
        counts["neutral_promotions"] += 1

    return {"trend": trend, "rule": rule, "counts": counts}


def run_model(features: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    t = config.get("thresholds", {}) or {}
    model_id = str(config.get("model_id", "trend_method_v2_2"))

    baseline = _base_v2_1_decision(features, t)
    base_trend = str(baseline.get("trend") or "Neutral")

    thresholds = {
        "reversal_pressure_high": _f(t.get("reversal_pressure_high", 0.08), 0.08),
        "snap_score_high": _f(t.get("snap_score_high", 0.33), 0.33),
        "snap_velocity_min": _f(t.get("snap_velocity_min", 0.0), 0.0),
        "hinge_torsion_high": _f(t.get("hinge_torsion_high", 0.35), 0.35),
        "hinge_velocity_min": _f(t.get("hinge_velocity_min", 0.001), 0.001),
        "fan_width_velocity_collapse": _f(t.get("fan_width_velocity_collapse", -0.005), -0.005),
        "fan_width_velocity_expand": _f(t.get("fan_width_velocity_expand", 0.005), 0.005),
        "fan_width_acceleration_collapse": _f(t.get("fan_width_acceleration_collapse", -0.002), -0.002),
        "phase_alignment_min": _f(t.get("phase_alignment_min", 0.60), 0.60),
        "fan_phase_score_min": _f(t.get("fan_phase_score_min", 0.20), 0.20),
        "slow_retention_high": _f(t.get("slow_retention_high", 0.03), 0.03),
        "outer_fan_width_min": _f(t.get("outer_fan_width_min", 0.01), 0.01),
        "neutral_promotion_min_strength": _f(t.get("neutral_promotion_min_strength", 0.0025), 0.0025),
        "false_snap_hinge_max": _f(t.get("false_snap_hinge_max", 0.20), 0.20),
        "true_snap_hinge_min": _f(t.get("true_snap_hinge_min", 0.32), 0.32),
    }

    fx = build_v2_2_features(features, baseline, thresholds)
    ruled = apply_v2_2_rules(base_trend, baseline, fx, thresholds)

    trend = ruled["trend"]
    confidence = min(1.0, max(0.0, (baseline["fan_width"] + abs(baseline["compression_velocity"]) + abs(baseline["flip_score"])) / 3.0))
    score = float(baseline["slope_sum"] if trend != "Neutral" else 0.0)

    return {
        "model_id": model_id,
        "trend": trend,
        "confidence": float(confidence),
        "score": score,
        "reason": f"regime={baseline['regime']}|slope_sum={baseline['slope_sum']:.6f}|v2_2_rule={ruled['rule']}",
        "debug": {
            "regime": baseline["regime"],
            "run_direction": baseline["run_direction"],
            "thresholds": {**{k: _f(v) for k, v in t.items()}, **thresholds},
            "signals": {
                "fan_width": baseline["fan_width"],
                "alignment": baseline["alignment"],
                "compression_velocity": baseline["compression_velocity"],
                "flip_score": baseline["flip_score"],
                "slope_sum": baseline["slope_sum"],
                "slope_magnitude": baseline["slope_magnitude"],
                "ratio_8_23_to_23_53": baseline["ratio_8_23_to_23_53"],
                "reversal_pressure": fx["reversal_pressure"],
            },
            "v2_1_replay": {
                "override_applied": baseline["v2_1_override"],
                "rule": "bull_run_instability_inversion",
            },
            "v2_2": {
                "base_trend": base_trend,
                "rule": ruled["rule"],
                "override_impact": ruled["counts"],
                "fan_features": fx,
            },
        },
        "raw_features_used": {
            "s23": _f((features.get("gauss") or {}).get("slopes", {}).get("s23"), 0.0),
            "s53": _f((features.get("gauss") or {}).get("slopes", {}).get("s53"), 0.0),
            "fan_width": baseline["fan_width"],
            "alignment": baseline["alignment"],
            "compression_velocity": baseline["compression_velocity"],
            "flip_score": baseline["flip_score"],
            "reversal_pressure": fx["reversal_pressure"],
            "hinge_velocity": fx["hinge_velocity"],
            "snap_velocity": fx["snap_velocity"],
            "fan_width_acceleration": fx["fan_width_acceleration"],
        },
    }