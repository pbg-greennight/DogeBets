from __future__ import annotations

from typing import Any, Mapping


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _pick(*values: Any, default: float = 0.0) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def build_v21_feature_vector(td_features: dict) -> dict:
    gauss = td_features.get("gauss", {})
    fan = td_features.get("fan", {})
    gcs = td_features.get("gcs", {})
    hyst = td_features.get("hysteresis", {})
    comp = td_features.get("compression", {})

    slopes = gauss.get("slopes", {})
    curvature = gauss.get("curvature", {})

    return {
        # --- CORE TREND SIGNAL ---
        "slope_s8": slopes.get("s8", 0.0),
        "slope_s23": slopes.get("s23", 0.0),
        "slope_s50": slopes.get("s50", 0.0),
        "slope_s83": slopes.get("s83", 0.0),

        "curvature_s23": curvature.get("s23", 0.0),
        "curvature_s50": curvature.get("s50", 0.0),

        # --- FAN STRUCTURE ---
        "fan_width": fan.get("width", 0.0),
        "fan_alignment": fan.get("alignment", 0.0),

        # --- COMPRESSION ---
        "compression": comp.get("value", 0.0),

        # --- GCS (CHANNEL CONTEXT) ---
        "gcs_position": gcs.get("position", 0.0),
        "gcs_slope": gcs.get("slope", 0.0),
        "gcs_width": gcs.get("width", 0.0),

        # --- HYSTERESIS ---
        "hyst_state": hyst.get("state", 0.0),
        "hyst_stability": hyst.get("stability", 0.0),
    }


def mapped_context_is_active(mapped: Mapping[str, Any]) -> bool:
    """
    Return True only when the richer non-slope context features are actually
    present/non-zero. This avoids treating simple slope passthrough values as
    proof that the richer mapped stack is active.
    """
    slope_keys = {
        "slope_s8",
        "slope_s23",
        "slope_s38",
        "slope_s53",
        "slope_s68",
        "slope_s83",
    }
    for key, value in dict(mapped or {}).items():
        if key in slope_keys:
            continue
        try:
            if float(value) != 0.0:
                return True
        except Exception:
            if value not in (None, False, "", [], {}):
                return True
    return False
