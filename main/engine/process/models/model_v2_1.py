from __future__ import annotations

from typing import Any, Dict


def _sign_label(v: float) -> str:
    if v > 0:
        return "Bull"
    if v < 0:
        return "Bear"
    return "Neutral"


def run_model(features: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    t = config.get("thresholds", {}) or {}
    model_id = str(config.get("model_id", "trend_method_v2_1"))

    gauss = (features.get("gauss") or {})
    fan = (features.get("fan") or {})
    spacing = (features.get("spacing") or {})
    compression = (features.get("compression") or {})
    hysteresis = (features.get("hysteresis") or {})
    torque = (features.get("torque") or {})

    slopes = (gauss.get("slopes") or {})
    s23 = float(slopes.get("s23", 0.0))
    s53 = float(slopes.get("s53", 0.0))
    slope_sum = s23 + s53
    slope_magnitude = abs(slope_sum)

    fan_width = float(fan.get("width_abs", fan.get("width", 0.0)))
    alignment = float(torque.get("alignment", 0.0))
    comp_vel = float(compression.get("velocity", 0.0))
    flip_score = float(hysteresis.get("flip_score", 0.0))
    ratio_8_23_to_23_53 = float(spacing.get("ratio_8_23_to_23_53", 0.0))

    run_fan_threshold = float(t.get("run_fan_threshold", 0.01))
    noise_fan_threshold = float(t.get("noise_fan_threshold", 0.004))
    compression_threshold = float(t.get("compression_threshold", 0.003))
    flip_score_threshold = float(t.get("flip_score_threshold", 0.5))
    slope_noise_threshold = float(t.get("slope_noise_threshold", 0.002))
    alignment_threshold = float(t.get("alignment_threshold", 0.5))

    bull_run_flip_score_threshold = float(t.get("bull_run_flip_score_threshold", 0.18))
    bull_run_compression_min = float(t.get("bull_run_compression_min", 0.0))
    bull_run_ratio_max = float(t.get("bull_run_ratio_max", -0.10))

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

    confidence = min(1.0, max(0.0, (fan_width + abs(comp_vel) + abs(flip_score)) / 3.0))
    score = float(slope_sum if trend != "Neutral" else 0.0)

    return {
        "model_id": model_id,
        "trend": trend,
        "confidence": float(confidence),
        "score": score,
        "reason": (
            f"regime={regime}|slope_sum={slope_sum:.6f}|"
            f"v2_1_override={'bull_run_instability' if bull_run_instability_override else 'none'}"
        ),
        "debug": {
            "regime": regime,
            "run_direction": run_direction,
            "thresholds": {
                "run_fan_threshold": run_fan_threshold,
                "noise_fan_threshold": noise_fan_threshold,
                "compression_threshold": compression_threshold,
                "flip_score_threshold": flip_score_threshold,
                "slope_noise_threshold": slope_noise_threshold,
                "alignment_threshold": alignment_threshold,
                "bull_run_flip_score_threshold": bull_run_flip_score_threshold,
                "bull_run_compression_min": bull_run_compression_min,
                "bull_run_ratio_max": bull_run_ratio_max,
            },
            "signals": {
                "fan_width": fan_width,
                "alignment": alignment,
                "compression_velocity": comp_vel,
                "flip_score": flip_score,
                "slope_sum": slope_sum,
                "slope_magnitude": slope_magnitude,
                "ratio_8_23_to_23_53": ratio_8_23_to_23_53,
            },
            "v2_1_replay": {
                "override_applied": bull_run_instability_override,
                "rule": "bull_run_instability_inversion",
            },
        },
        "raw_features_used": {
            "s23": s23,
            "s53": s53,
            "fan_width": fan_width,
            "alignment": alignment,
            "compression_velocity": comp_vel,
            "flip_score": flip_score,
        },
    }
