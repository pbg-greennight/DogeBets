# main/engine/process/models/model_v1_0.py

from __future__ import annotations
from typing import Any, Dict


def _trend_from_score(score: float, threshold: float) -> str:
    if score > threshold:
        return "Bull"
    if score < -threshold:
        return "Bear"
    return "Neutral"


def run_model(features: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    t = config.get("thresholds", {}) or {}
    model_id = str(config.get("model_id", "trend_method_unknown"))

    slopes = (features.get("gauss", {}) or {}).get("slopes", {}) or {}
    spacing = (features.get("spacing", {}) or {})
    fan = (features.get("fan", {}) or {})
    compression = (features.get("compression", {}) or {})
    hysteresis = (features.get("hysteresis", {}) or {})
    torque = (features.get("torque", {}) or {})

    slope_core = float(slopes.get("s23", 0.0)) + float(slopes.get("s53", 0.0))
    fan_width = float(fan.get("width", 0.0))
    ratio = float(spacing.get("ratio_8_23_to_23_53", 0.0))
    flip_score = float(hysteresis.get("flip_score", 0.0))
    alignment = float(torque.get("alignment", 0.0))
    comp_vel = float(compression.get("velocity", 0.0))

    # versioned score weights
    suffix = model_id.split("_")[-1]
    weights = {
        "v1_0": (1.00, 0.20, 0.00, 0.00, 0.00),
        "v1_1": (0.90, 0.25, 0.10, 0.00, 0.00),
        "v1_2": (0.80, 0.30, 0.10, 0.15, 0.00),
        "v1_3": (0.75, 0.35, 0.15, 0.10, 0.20),
        "v1_4": (0.70, 0.40, 0.20, 0.15, 0.25),
        "v1_5": (0.65, 0.45, 0.25, 0.20, 0.30),
    }
    w1, w2, w3, w4, w5 = weights.get(suffix, weights["v1_0"])

    score = (w1 * slope_core) + (w2 * fan_width) + (w3 * ratio) + (w4 * comp_vel) + (w5 * alignment)
    score -= float(t.get("flip_penalty", 0.05)) * flip_score

    threshold = float(t.get("score_threshold", 0.02))
    trend = _trend_from_score(score, threshold)

    return {
        "model_id": model_id,
        "trend": trend,
        "confidence": 1.0,
        "score": float(score),
        "reason": f"score_vs_threshold:{score:.6f}/{threshold:.6f}",
        "debug": {
            "weights": {"slope": w1, "fan": w2, "ratio": w3, "compression": w4, "torque": w5},
            "threshold": threshold,
            "flip_penalty": float(t.get("flip_penalty", 0.05)),
        },
        "raw_features_used": {
            "s8": float(((features.get("gauss", {}) or {}).get("latest", {}) or {}).get("s8", 0.0)),
            "s23": float(((features.get("gauss", {}) or {}).get("latest", {}) or {}).get("s23", 0.0)),
            "s53": float(((features.get("gauss", {}) or {}).get("latest", {}) or {}).get("s53", 0.0)),
            "fan_width": fan_width,
            "R1": ratio,
            "R2": float(spacing.get("ratio_53_83_to_23_53", 0.0)),
            "flip_score": flip_score,
        },
    }
