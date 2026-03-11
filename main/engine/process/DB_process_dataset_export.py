from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY_PATH = (SCRIPT_DIR / "models" / "model_predictions_history" / "model_predictions_history.jsonl").resolve()
DEFAULT_REPLAY_HISTORY_PATH = (SCRIPT_DIR / "models" / "model_predictions_history" / "model_predictions_history_with_v2_2.jsonl").resolve()
DEFAULT_ROUND_PATH = (SCRIPT_DIR / "../ts/json/round_record.json").resolve()
DEFAULT_OUTPUT_DIR = (SCRIPT_DIR / "datasets").resolve()

# Tunable analysis constants.
COLLAPSE_COMPRESSION_THRESHOLD = -0.8
WEAK_ALIGNMENT_THRESHOLD = 0.25
FLIP_SCORE_ALERT_THRESHOLD = 0.05
RUN_ALIGNMENT_MIN = 0.50
NOISE_SLOPE_MAX = 0.002
FLIP_ZONE_RATIO_THRESHOLD = 0.0
EPSILON = 1e-9
HINGE_TORSION_SPIKE_THRESHOLD = 0.8
FAN_COLLAPSE_V2_VELOCITY_THRESHOLD = -20.0
FAN_COLLAPSE_V2_STRICT_VELOCITY_THRESHOLD = -40.0
COMPRESSION_TRAP_VELOCITY_THRESHOLD = -10.0
COMPRESSION_TRAP_STRICT_VELOCITY_THRESHOLD = -20.0
INERTIA_WALL_HIGH_THRESHOLD = 0.75

MODEL_V2_0 = "trend_method_v2_0"
MODEL_V2_1 = "trend_method_v2_1"
MODEL_V2_2 = "trend_method_v2_2"
MODEL_DEFAULT = MODEL_V2_0

CSV_COLUMNS = [
    "model_id",
    "epoch",
    "next_epoch",
    "timestamp",
    "prediction",
    "confidence",
    "score",
    "reason",
    "truth",
    "start_price",
    "end_price",
    "price_difference",
    "round_current_timestamp",
    "round_next_epoch_time",
    "correct",
    "directional_called",
    "directional_correct",
    "bull_called",
    "bear_called",
    "neutral_called",
    "bull_called_bear_truth",
    "bear_called_bull_truth",
    "neutral_called_bull_truth",
    "neutral_called_bear_truth",
    "bull_called_neutral_truth",
    "bear_called_neutral_truth",
    "regime",
    "run_direction",
    "g8",
    "g23",
    "g38",
    "g53",
    "g68",
    "g83",
    "slope_g8",
    "slope_g23",
    "slope_g38",
    "slope_g53",
    "slope_g68",
    "slope_g83",
    "fan_width",
    "fan_width_velocity",
    "fan_width_acceleration",
    "fan_order_score",
    "g83_curvature",
    "alignment",
    "compression_velocity",
    "flip_score",
    "slope_sum",
    "slope_magnitude",
    "ratio_8_23_to_23_53",
    "s23",
    "s53",
    "fan_width_raw",
    "alignment_raw",
    "compression_velocity_raw",
    "flip_score_raw",
    "slope_sign_s23",
    "slope_sign_s53",
    "slope_disagreement",
    "slope_gap",
    "abs_slope_sum",
    "abs_compression_velocity",
    "abs_flip_score",
    "fan_to_slope_ratio",
    "collapse_flag",
    "weak_bull_exhaustion_flag",
    "weak_bear_exhaustion_flag",
    "possible_flip_zone",
    "possible_run_zone",
    "possible_transition_zone",
    "fan_energy_total",
    "fan_energy_ratio",
    "fan_energy_velocity",
    "fan_energy_acceleration",
    "fan_sign_conflict",
    "fan_energy_instability",
    "hinge_gap",
    "hinge_velocity",
    "hinge_conflict",
    "hinge_torsion",
    "snap_divergence",
    "snap_velocity",
    "snap_score",
    "phase_alignment",
    "phase_strength",
    "phase_velocity",
    "fan_phase_score",
    "phase_disagreement",
    "hinge_torsion_spike",
    "transition_score_base",
    "fan_polarity_inversion",
    "fan_polarity_normal",
    "outer_fan_width",
    "outer_fan_width_prev",
    "outer_order_signature",
    "slow_retention",
    "gaussian_wave",
    "wave_strength",
    "fan_velocity_sign_t0",
    "fan_velocity_sign_t1",
    "fan_velocity_sign_t2",
    "fan_velocity_pattern_3",
    "fan_velocity_oscillation",
    "fan_velocity_oscillation_strength",
    "fan_collapse_v2",
    "fan_collapse_v2_strict",
    "compression_trap",
    "compression_trap_strict",
    "epoch_start_price",
    "epoch_end_price",
    "epoch_price_diff",
    "epoch_return",
    "epoch_volatility",
    "price_diff_t0",
    "price_diff_t1",
    "price_diff_t2",
    "momentum_3epoch",
    "price_diff_3epoch_abs_sum",
    "price_diff_sign_pattern",
    "direction_flip",
    "price_diff_decay_ratio",
    "price_diff_vol_squeeze",
    "fan_energy_t0",
    "fan_energy_t1",
    "fan_energy_t2",
    "fan_energy_t3",
    "fan_energy_release_signal",
    "fan_energy_cycle_stage",
    "inertia_wall_score",
    "reversal_pressure",
    "tail_anchor_type",
    "extrema_pair",
    "v2_2_rule",
    "directional_error_type",
    "truth_direction_group",
    "prediction_direction_group",
]

log = logging.getLogger("DB_process_dataset_export")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%I:%M:%S %p")

def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("$", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _norm_direction(label: Any) -> str:
    txt = str(label or "").strip().lower()
    if txt in {"bull", "bullish", "up", "long", "+1", "1"}:
        return "Bull"
    if txt in {"bear", "bearish", "down", "short", "-1"}:
        return "Bear"
    if txt in {"neutral", "flat", "0", "none", "abstain", ""}:
        return "Neutral"
    return "Unknown"


def _truth_from_price_diff(price_diff: Optional[float]) -> str:
    if price_diff is None:
        return "Unknown"
    if price_diff > 0:
        return "Bull"
    if price_diff < 0:
        return "Bear"
    return "Neutral"


def _sign(value: Optional[float]) -> int:
    if value is None:
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0

def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _safe_mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [v for v in (_as_float(r.get(key)) for r in rows) if v is not None]
    return float(mean(vals)) if vals else 0.0


def _normalize_regime(value: Any) -> str:
    txt = str(value or "").strip().upper()
    if txt in {"RUN", "REVERSAL", "NOISE"}:
        return txt
    return "UNKNOWN"

def compute_gaussian_wave(velocity_window: Sequence[float]) -> Optional[int]:
    """Compute baseline gaussian-wave pattern from the latest 3 valid fan width velocities."""
    if len(velocity_window) < 3:
        return None
    v2, v1, v0 = velocity_window[-3], velocity_window[-2], velocity_window[-1]
    return 1 if (v2 > 0 and v1 < 0 and v0 > 0) else 0


def compute_wave_strength(velocity_window: Sequence[float]) -> Optional[float]:
    """Compute oscillation magnitude from the latest 3 valid fan width velocities."""
    if len(velocity_window) < 3:
        return None
    v2, v1, v0 = velocity_window[-3], velocity_window[-2], velocity_window[-1]
    return abs(v2) + abs(v1) + abs(v0)


def compute_slow_retention(row: Dict[str, Any]) -> Optional[float]:
    """Sum absolute slow-stack slopes over valid values only (g53/g68/g83)."""
    slow_slopes = [_as_float(row.get("slope_g53")), _as_float(row.get("slope_g68")), _as_float(row.get("slope_g83"))]
    valid = [v for v in slow_slopes if v is not None]
    if not valid:
        return None
    return sum(abs(v) for v in valid)


def compute_outer_fan_width(row: Dict[str, Any]) -> Optional[float]:
    """Compute absolute spacing between outer slow-fan levels g83 and g53."""
    g53 = _as_float(row.get("g53"))
    g83 = _as_float(row.get("g83"))
    if None in (g53, g83):
        return None
    return abs(g83 - g53)


def compute_phase_disagreement(row: Dict[str, Any]) -> Optional[int]:
    """Detect fast/slow phase-sign disagreement via mean slope signs."""
    fast = [
        _as_float(row.get("slope_g8")),
        _as_float(row.get("slope_g23")),
        _as_float(row.get("slope_g38")),
    ]
    slow = [
        _as_float(row.get("slope_g53")),
        _as_float(row.get("slope_g68")),
        _as_float(row.get("slope_g83")),
    ]
    fast_valid = [v for v in fast if v is not None]
    slow_valid = [v for v in slow if v is not None]
    if not fast_valid or not slow_valid:
        return None
    fast_sign = _sign(sum(fast_valid) / len(fast_valid))
    slow_sign = _sign(sum(slow_valid) / len(slow_valid))
    if fast_sign == 0 or slow_sign == 0:
        return None
    return 1 if fast_sign != slow_sign else 0


def compute_inertia_wall_score(row: Dict[str, Any]) -> Optional[float]:
    """Blend wave/inertia terms using normalized valid weights only."""
    terms = [
        (0.40, _as_float(row.get("slow_retention"))),
        (0.25, _as_float(row.get("outer_fan_width"))),
        (0.20, _as_float(row.get("phase_disagreement"))),
        (0.15, _as_float(row.get("hinge_torsion"))),
    ]
    weighted_sum = 0.0
    valid_weight_sum = 0.0
    for weight, value in terms:
        if value is None:
            continue
        weighted_sum += weight * value
        valid_weight_sum += weight
    if valid_weight_sum <= 0.0:
        return None
    return weighted_sum / valid_weight_sum


def build_wave_inertia_profile(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build class-wise wave/inertia feature means and non-null counts."""
    classes = [
        "correct_bull",
        "correct_bear",
        "bull_called_bear_truth",
        "bear_called_bull_truth",
        "neutral_called_bull_truth",
        "neutral_called_bear_truth",
    ]
    features = [
        "gaussian_wave",
        "wave_strength",
        "slow_retention",
        "outer_fan_width",
        "phase_disagreement",
        "inertia_wall_score",
    ]
    out: List[Dict[str, Any]] = []
    for label in classes:
        subset = [r for r in rows if r.get("directional_error_type") == label]
        profile: Dict[str, Any] = {
            "directional_error_type": label,
            "row_count": len(subset),
            "coverage_within_export_set": _safe_ratio(len(subset), len(rows)),
        }
        for feat in features:
            values = [_as_float(r.get(feat)) for r in subset]
            valid = [v for v in values if v is not None]
            profile[f"mean_{feat}"] = float(mean(valid)) if valid else None
            profile[f"count_{feat}"] = len(valid)
        out.append(profile)
    return out

def _read_nested_float(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    """Return first parseable numeric field from a dict for candidate keys."""
    for key in keys:
        if key in payload:
            value = _as_float(payload.get(key))
            if value is not None:
                return value
    return None

def _resolve_sort_key(row: Dict[str, Any]) -> Tuple[int, int, str]:
    """Stable chronological ordering key used for temporal derivatives."""
    epoch = _as_int(row.get("epoch"))
    next_epoch = _as_int(row.get("next_epoch"))
    timestamp = str(row.get("timestamp") or "")
    return (
        epoch if epoch is not None else 10**18,
        next_epoch if next_epoch is not None else 10**18,
        timestamp,
    )

def _extract_sigma_level_features(model: Dict[str, Any], debug: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract sigma values/slopes from known history schemas, leaving missing values as None."""
    signals = debug.get("signals") if isinstance(debug.get("signals"), dict) else {}
    # History payloads vary by run; fields can appear at top level, debug blocks,
    # or nested sigma containers under signals/raw_features_used.
    sigma_block_candidates = [
        signals.get("gaussian_levels"),
        signals.get("sigma_levels"),
        signals.get("gaussian_fan"),
        raw.get("gaussian_levels"),
        raw.get("sigma_levels"),
        raw.get("gaussian_fan"),
        debug.get("gaussian_levels"),
        debug.get("sigma_levels"),
        model.get("gaussian_levels"),
        model.get("sigma_levels"),
    ]
    slope_block_candidates = [
        signals.get("gaussian_slopes"),
        signals.get("sigma_slopes"),
        raw.get("gaussian_slopes"),
        raw.get("sigma_slopes"),
        debug.get("gaussian_slopes"),
        debug.get("sigma_slopes"),
        model.get("gaussian_slopes"),
        model.get("sigma_slopes"),
    ]
    sigma_sources = [signals, raw, debug, model] + [s for s in sigma_block_candidates if isinstance(s, dict)] + [s for s in slope_block_candidates if isinstance(s, dict)]

    sigma_aliases: Dict[str, Sequence[str]] = {
        "g8": ("g8", "gauss8", "sigma8", "sigma_8"),
        "g23": ("g23", "gauss23", "sigma23", "sigma_23"),
        "g38": ("g38", "gauss38", "sigma38", "sigma_38"),
        "g53": ("g53", "gauss53", "sigma53", "sigma_53"),
        "g68": ("g68", "gauss68", "sigma68", "sigma_68"),
        "g83": ("g83", "gauss83", "sigma83", "sigma_83"),
    }
    slope_aliases: Dict[str, Sequence[str]] = {
        "slope_g8": ("slope_g8", "g8_slope", "s8", "slope_8"),
        "slope_g23": ("slope_g23", "g23_slope", "s23", "slope_23"),
        "slope_g38": ("slope_g38", "g38_slope", "s38", "slope_38"),
        "slope_g53": ("slope_g53", "g53_slope", "s53", "slope_53"),
        "slope_g68": ("slope_g68", "g68_slope", "s68", "slope_68"),
        "slope_g83": ("slope_g83", "g83_slope", "s83", "slope_83"),
    }

    extracted: Dict[str, Any] = {}
    for out_key, aliases in sigma_aliases.items():
        value = None
        for source in sigma_sources:
            value = _read_nested_float(source, aliases)
            if value is not None:
                break
        extracted[out_key] = value

    for out_key, aliases in slope_aliases.items():
        value = None
        for source in sigma_sources:
            value = _read_nested_float(source, aliases)
            if value is not None:
                break
        extracted[out_key] = value

    return extracted

def extract_sigma_fields(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Public helper for extracting sigma values/slopes from joined model rows."""
    debug = row.get("debug") if isinstance(row.get("debug"), dict) else {}
    raw = row.get("raw_features_used") if isinstance(row.get("raw_features_used"), dict) else {}
    return _extract_sigma_level_features(row, debug, raw)


def compute_geometry_features(row: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Compute geometry-family features from sigma-level values only when inputs are valid."""
    g8, g23, g38, g53, g68, g83 = (_as_float(row.get(k)) for k in ["g8", "g23", "g38", "g53", "g68", "g83"])
    slope_sum = _as_float(row.get("slope_sum"))
    fan_width = (g83 - g8) if None not in (g83, g8) else None
    prev_fan_width = _as_float((prev or {}).get("fan_width"))
    fan_width_velocity = fan_width - prev_fan_width if None not in (fan_width, prev_fan_width) else None
    prev_fan_width_velocity = _as_float((prev or {}).get("fan_width_velocity"))
    fan_width_acceleration = (
        fan_width_velocity - prev_fan_width_velocity
        if None not in (fan_width_velocity, prev_fan_width_velocity)
        else None
    )
    stack = [g8, g23, g38, g53, g68, g83]
    fan_order_score = _ordered_score(stack, _sign(slope_sum) if _sign(slope_sum) != 0 else 1)
    prev_g83 = _as_float((prev or {}).get("g83"))
    prev_prev_g83 = _as_float((prev or {}).get("prev_g83"))
    g83_curvature = (g83 - 2 * prev_g83 + prev_prev_g83) if None not in (g83, prev_g83, prev_prev_g83) else None
    return {
        "fan_width": fan_width,
        "fan_width_velocity": fan_width_velocity,
        "fan_width_acceleration": fan_width_acceleration,
        "fan_order_score": fan_order_score,
        "g83_curvature": g83_curvature,
    }


def compute_energy_features(row: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute energy-family features from slope fields with explicit missing handling."""
    slope_values = [_as_float(row.get(k)) for k in ["slope_g8", "slope_g23", "slope_g38", "slope_g53", "slope_g68", "slope_g83"]]
    valid_slopes = [x for x in slope_values if x is not None]
    fan_energy_total = sum(abs(x) for x in valid_slopes) if valid_slopes else None
    inner = [x for x in slope_values[:3] if x is not None]
    outer = [x for x in slope_values[3:] if x is not None]
    fan_energy_ratio = (sum(abs(x) for x in inner) / sum(abs(x) for x in outer)) if inner and outer and sum(abs(x) for x in outer) > 0 else None
    prev_energy = _as_float((prev or {}).get("fan_energy_total"))
    fan_energy_velocity = fan_energy_total - prev_energy if None not in (fan_energy_total, prev_energy) else None
    prev_energy_velocity = _as_float((prev or {}).get("fan_energy_velocity"))
    fan_energy_acceleration = (
        fan_energy_velocity - prev_energy_velocity
        if None not in (fan_energy_velocity, prev_energy_velocity)
        else None
    )
    sign_g8 = _phase_sign(_as_float(row.get("slope_g8")))
    sign_g83 = _phase_sign(_as_float(row.get("slope_g83")))
    fan_sign_conflict = bool(sign_g8 is not None and sign_g83 is not None and sign_g8 != sign_g83)
    fan_energy_instability = None
    if fan_energy_ratio is not None and fan_energy_acceleration is not None:
        normalized_ratio = min(max(fan_energy_ratio / (1.0 + fan_energy_ratio), 0.0), 1.0)
        negative_accel = min(max(-fan_energy_acceleration, 0.0), 1.0)
        fan_energy_instability = 0.4 * normalized_ratio + 0.3 * (1.0 if fan_sign_conflict else 0.0) + 0.3 * negative_accel
    return {
        "fan_energy_total": fan_energy_total,
        "fan_energy_ratio": fan_energy_ratio,
        "fan_energy_velocity": fan_energy_velocity,
        "fan_energy_acceleration": fan_energy_acceleration,
        "fan_sign_conflict": fan_sign_conflict,
        "fan_energy_instability": fan_energy_instability,
    }


def compute_hinge_features(row: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute hinge-family features when g23/g38 and slopes are available."""
    g23 = _as_float(row.get("g23"))
    g38 = _as_float(row.get("g38"))
    slope_g23 = _as_float(row.get("slope_g23"))
    slope_g38 = _as_float(row.get("slope_g38"))
    hinge_gap = (g23 - g38) if None not in (g23, g38) else None
    prev_hinge_gap = _as_float((prev or {}).get("hinge_gap"))
    hinge_velocity = (hinge_gap - prev_hinge_gap) if None not in (hinge_gap, prev_hinge_gap) else None
    sign_g23 = _phase_sign(slope_g23)
    sign_g38 = _phase_sign(slope_g38)
    hinge_conflict = bool(sign_g23 is not None and sign_g38 is not None and sign_g23 != sign_g38)
    hinge_torsion = abs(slope_g23 - slope_g38) / (abs(slope_g23) + abs(slope_g38)) if None not in (slope_g23, slope_g38) and (abs(slope_g23) + abs(slope_g38)) > 0 else None
    return {
        "hinge_gap": hinge_gap,
        "hinge_velocity": hinge_velocity,
        "hinge_conflict": hinge_conflict,
        "hinge_torsion": hinge_torsion,
    }


def compute_snap_features(row: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Compute snap-family features only when inner fan slopes are all present."""
    slope_g8 = _as_float(row.get("slope_g8"))
    slope_g23 = _as_float(row.get("slope_g23"))
    slope_g38 = _as_float(row.get("slope_g38"))
    snap_divergence = None
    if None not in (slope_g8, slope_g23, slope_g38):
        snap_divergence = abs(slope_g8 - slope_g23) + abs(slope_g23 - slope_g38)
    prev_snap_divergence = _as_float((prev or {}).get("snap_divergence"))
    snap_velocity = (snap_divergence - prev_snap_divergence) if None not in (snap_divergence, prev_snap_divergence) else None
    snap_score = None
    if None not in (slope_g8, slope_g23, slope_g38):
        denom = abs(slope_g8) + abs(slope_g23) + abs(slope_g38)
        if denom > 0 and snap_divergence is not None:
            snap_score = snap_divergence / denom
    return {"snap_divergence": snap_divergence, "snap_velocity": snap_velocity, "snap_score": snap_score}


def compute_phase_features(row: Dict[str, Any], prev: Optional[Dict[str, Any]], rolling_phase_strength_max: float) -> Dict[str, Optional[float]]:
    """Compute phase-family features from slope sign coherence with minimum input coverage."""
    slope_values = [_as_float(row.get(k)) for k in ["slope_g8", "slope_g23", "slope_g38", "slope_g53", "slope_g68", "slope_g83"]]
    signs = [s for s in [_phase_sign(x) for x in slope_values] if s is not None]
    phase_alignment = None
    if len(signs) >= 4:
        pos = sum(1 for s in signs if s > 0)
        neg = sum(1 for s in signs if s < 0)
        phase_alignment = max(pos, neg) / len(signs)
    phase_strength = _as_float(row.get("fan_energy_total"))
    prev_phase_alignment = _as_float((prev or {}).get("phase_alignment"))
    phase_velocity = (phase_alignment - prev_phase_alignment) if None not in (phase_alignment, prev_phase_alignment) else None
    fan_phase_score = None
    if phase_alignment is not None and phase_strength is not None and rolling_phase_strength_max > 0:
        fan_phase_score = phase_alignment * (phase_strength / rolling_phase_strength_max)
    return {
        "phase_alignment": phase_alignment,
        "phase_strength": phase_strength,
        "phase_velocity": phase_velocity,
        "fan_phase_score": fan_phase_score,
        "phase_input_count": len(signs),
    }


def _ordered_score(values: Sequence[Optional[float]], direction: int) -> Optional[float]:
    pairs = [(values[idx], values[idx + 1]) for idx in range(len(values) - 1)]
    valid_pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not valid_pairs:
        return None
    if direction >= 0:
        correct = sum(1 for a, b in valid_pairs if a > b)
    else:
        correct = sum(1 for a, b in valid_pairs if a < b)
    return correct / len(valid_pairs)


def _phase_sign(value: Optional[float]) -> Optional[int]:
    if value is None or value == 0:
        return None
    return 1 if value > 0 else -1

def _sign_to_char(value: Optional[float]) -> str:
    s = _sign(value)
    if s > 0:
        return "+"
    if s < 0:
        return "-"
    return "0"


def _is_transition_reversal(row: Dict[str, Any]) -> bool:
    truth = str(row.get("truth") or "")
    run_direction = str(row.get("run_direction") or "")
    if truth not in {"Bull", "Bear"}:
        return False
    if run_direction not in {"Bull", "Bear"}:
        return False
    return truth != run_direction


def _resolve_directional_prediction(row: Dict[str, Any], key: str) -> str:
    pred = str(row.get(key) or "")
    if pred in {"Bull", "Bear", "Neutral"}:
        return pred
    return "Neutral"

def _extract_v2_0_override_context(model_row: Dict[str, Any]) -> Dict[str, Any]:
    debug = model_row.get("debug") if isinstance(model_row.get("debug"), dict) else {}
    signals = debug.get("signals") if isinstance(debug.get("signals"), dict) else {}
    return {
        "regime": str(debug.get("regime") or ""),
        "run_direction": str(debug.get("run_direction") or ""),
        "flip_score": _as_float(signals.get("flip_score")),
        "compression_velocity": _as_float(signals.get("compression_velocity")),
        "ratio_8_23_to_23_53": _as_float(signals.get("ratio_8_23_to_23_53")),
    }


def build_replay_history_with_v2_1(
    history_rows: Sequence[Dict[str, Any]],
    v2_1_thresholds: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    replay_rows: List[Dict[str, Any]] = []
    stats = {
        "rows_seen": 0,
        "rows_with_models": 0,
        "rows_with_v2_0": 0,
        "rows_with_existing_v2_1": 0,
        "rows_with_v2_1_appended": 0,
        "rows_skipped_missing_signals": 0,
        "bull_to_bear_overrides": 0,
    }

    flip_thr = float(v2_1_thresholds.get("bull_run_flip_score_threshold", 0.18))
    comp_thr = float(v2_1_thresholds.get("bull_run_compression_min", 0.0))
    ratio_thr = float(v2_1_thresholds.get("bull_run_ratio_max", -0.10))

    for row in history_rows:
        stats["rows_seen"] += 1
        if not isinstance(row, dict):
            continue

        new_row = dict(row)
        models = row.get("models") if isinstance(row.get("models"), list) else None
        if models is None:
            replay_rows.append(new_row)
            continue

        stats["rows_with_models"] += 1
        new_models = list(models)

        if any(str(m.get("model_id") or "") == MODEL_V2_1 for m in models if isinstance(m, dict)):
            stats["rows_with_existing_v2_1"] += 1
            new_row["models"] = new_models
            replay_rows.append(new_row)
            continue

        v2_0 = next((m for m in models if isinstance(m, dict) and str(m.get("model_id") or "") == MODEL_V2_0), None)
        if v2_0 is None:
            new_row["models"] = new_models
            replay_rows.append(new_row)
            continue

        stats["rows_with_v2_0"] += 1
        ctx = _extract_v2_0_override_context(v2_0)
        trend = _norm_direction(v2_0.get("trend"))

        required = [ctx.get("flip_score"), ctx.get("compression_velocity"), ctx.get("ratio_8_23_to_23_53")]
        if any(v is None for v in required):
            stats["rows_skipped_missing_signals"] += 1
            v2_1 = dict(v2_0)
            v2_1["model_id"] = MODEL_V2_1
            v2_1["reason"] = f"{v2_0.get('reason')}|v2_1_replay=insufficient_signals"
        else:
            override = (
                trend == "Bull"
                and str(ctx.get("regime") or "").upper() == "RUN"
                and float(ctx["flip_score"]) >= flip_thr
                and float(ctx["compression_velocity"]) >= comp_thr
                and float(ctx["ratio_8_23_to_23_53"]) <= ratio_thr
            )
            new_trend = "Bear" if override else trend
            if override:
                stats["bull_to_bear_overrides"] += 1
            v2_1 = dict(v2_0)
            v2_1["model_id"] = MODEL_V2_1
            v2_1["trend"] = new_trend
            v2_1["reason"] = (
                f"{v2_0.get('reason')}|v2_1_override={'bull_run_instability' if override else 'none'}"
            )
            debug = dict(v2_0.get("debug") or {})
            replay_debug = dict(debug.get("v2_1_replay") or {})
            replay_debug.update({
                "override_applied": bool(override),
                "source_model_id": MODEL_V2_0,
                "rule": "bull_run_instability_inversion",
                "thresholds": {
                    "bull_run_flip_score_threshold": flip_thr,
                    "bull_run_compression_min": comp_thr,
                    "bull_run_ratio_max": ratio_thr,
                },
            })
            debug["v2_1_replay"] = replay_debug
            v2_1["debug"] = debug

        new_models.append(v2_1)
        new_row["models"] = new_models
        stats["rows_with_v2_1_appended"] += 1
        replay_rows.append(new_row)

    return replay_rows, stats

def _build_features_from_history_model(model_row: Dict[str, Any], history_row: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct model feature payload from stored history/debug fields for deterministic replay."""
    debug = model_row.get("debug") if isinstance(model_row.get("debug"), dict) else {}
    signals = debug.get("signals") if isinstance(debug.get("signals"), dict) else {}
    stack = debug.get("gaussian_stack") if isinstance(debug.get("gaussian_stack"), dict) else {}
    slopes = debug.get("gaussian_slopes") if isinstance(debug.get("gaussian_slopes"), dict) else {}
    physics = debug.get("fan_physics") if isinstance(debug.get("fan_physics"), dict) else {}
    context = debug.get("context") if isinstance(debug.get("context"), dict) else {}
    raw = model_row.get("raw_features_used") if isinstance(model_row.get("raw_features_used"), dict) else {}

    return {
        "meta": {
            "curr_epoch": _as_int(history_row.get("epoch")) or 0,
            "next_epoch": _as_int(history_row.get("next_epoch")) or 0,
            "timestamp": str(history_row.get("timestamp") or ""),
        },
        "gauss": {
            "latest": {
                "s8": _as_float(stack.get("g8")) or 0.0,
                "s23": _as_float(stack.get("g23")) or 0.0,
                "s38": _as_float(stack.get("g38")) or 0.0,
                "s53": _as_float(stack.get("g53")) or 0.0,
                "s68": _as_float(stack.get("g68")) or 0.0,
                "s83": _as_float(stack.get("g83")) or 0.0,
            },
            "slopes": {
                "s8": _as_float(slopes.get("slope_g8")) or 0.0,
                "s23": _as_float(slopes.get("slope_g23")) or (_as_float(raw.get("s23")) or 0.0),
                "s38": _as_float(slopes.get("slope_g38")) or 0.0,
                "s53": _as_float(slopes.get("slope_g53")) or (_as_float(raw.get("s53")) or 0.0),
                "s68": _as_float(slopes.get("slope_g68")) or 0.0,
                "s83": _as_float(slopes.get("slope_g83")) or 0.0,
            },
        },
        "fan": {
            "width": _as_float(signals.get("fan_width")) or (_as_float(raw.get("fan_width")) or 0.0),
            "width_abs": abs(_as_float(signals.get("fan_width")) or (_as_float(raw.get("fan_width")) or 0.0)),
            "width_velocity": _as_float(((physics.get("geometry") or {}).get("fan_width_velocity"))) or 0.0,
            "width_acceleration": _as_float(((physics.get("geometry") or {}).get("fan_width_acceleration"))) or 0.0,
            "hinge_velocity": _as_float(((physics.get("hinge") or {}).get("hinge_velocity"))) or 0.0,
            "snap_velocity": _as_float(((physics.get("snap") or {}).get("snap_velocity"))) or 0.0,
        },
        "spacing": {
            "ratio_8_23_to_23_53": _as_float(signals.get("ratio_8_23_to_23_53")) or 0.0,
        },
        "compression": {
            "velocity": _as_float(signals.get("compression_velocity")) or (_as_float(raw.get("compression_velocity")) or 0.0),
        },
        "hysteresis": {
            "flip_score": _as_float(signals.get("flip_score")) or (_as_float(raw.get("flip_score")) or 0.0),
        },
        "torque": {
            "alignment": _as_float(signals.get("alignment")) or (_as_float(raw.get("alignment")) or 0.0),
        },
        "history": {
            "fan_physics": physics,
            "context": context,
        },
        "context": {
            "tail_anchor_type": context.get("tail_anchor_type"),
            "extrema_pair": context.get("extrema_pair"),
        },
    }

def _merge_dict_deep(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge update keys into base dictionary in place."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict_deep(base[key], value)
            continue
        base[key] = deepcopy(value)
    return base


def _build_v2_2_replay_block(base_model: Dict[str, Any], v2_2_output: Dict[str, Any]) -> Dict[str, Any]:
    """Build replayed v2_2 block by inheriting the full base model payload."""
    replay_model = deepcopy(base_model)
    replay_model["model_id"] = MODEL_V2_2

    for key in ["trend", "confidence", "score", "reason", "notes"]:
        if key in v2_2_output:
            replay_model[key] = deepcopy(v2_2_output.get(key))

    base_debug = replay_model.get("debug") if isinstance(replay_model.get("debug"), dict) else {}
    new_debug = v2_2_output.get("debug") if isinstance(v2_2_output.get("debug"), dict) else {}
    replay_model["debug"] = _merge_dict_deep(base_debug, new_debug)

    base_raw = replay_model.get("raw_features_used") if isinstance(replay_model.get("raw_features_used"), dict) else {}
    new_raw = v2_2_output.get("raw_features_used") if isinstance(v2_2_output.get("raw_features_used"), dict) else {}
    replay_model["raw_features_used"] = _merge_dict_deep(base_raw, new_raw)

    return replay_model


def _replay_payload_health(model_row: Dict[str, Any]) -> str:
    """Compact replay payload health summary for v2_2 appended rows."""
    debug = model_row.get("debug") if isinstance(model_row.get("debug"), dict) else {}
    stack = debug.get("gaussian_stack") if isinstance(debug.get("gaussian_stack"), dict) else {}
    slopes = debug.get("gaussian_slopes") if isinstance(debug.get("gaussian_slopes"), dict) else {}
    fan = debug.get("fan_physics") if isinstance(debug.get("fan_physics"), dict) else {}
    context = debug.get("context") if isinstance(debug.get("context"), dict) else {}

    stack_keys = ["g8", "g23", "g38", "g53", "g68", "g83"]
    slope_keys = ["slope_g8", "slope_g23", "slope_g38", "slope_g53", "slope_g68", "slope_g83"]
    stack_present = sum(1 for k in stack_keys if _as_float(stack.get(k)) is not None)
    slope_present = sum(1 for k in slope_keys if _as_float(slopes.get(k)) is not None)
    fan_ok = "ok" if fan else "missing"
    context_ok = "ok" if context else "missing"
    return f"stack={stack_present}/6 slopes={slope_present}/6 fan={fan_ok} context={context_ok}"

def build_replay_history_with_v2_2(
    history_rows: Sequence[Dict[str, Any]],
    v2_2_config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    try:
        from main.engine.process.models import model_v2_2
    except Exception:
        import importlib.util
        model_path = SCRIPT_DIR / "models" / "model_v2_2.py"
        spec = importlib.util.spec_from_file_location("model_v2_2", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load model_v2_2 from {model_path}")
        model_v2_2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_v2_2)

    replay_rows: List[Dict[str, Any]] = []
    stats = {
        "rows_seen": 0,
        "rows_with_models": 0,
        "rows_with_v2_1": 0,
        "rows_with_existing_v2_2": 0,
        "rows_with_v2_2_appended": 0,
        "rows_skipped_missing_base": 0,
        "continuation_protection_blocks": 0,
        "collapse_reversal_overrides": 0,
        "false_snap_blocks": 0,
        "inertia_wall_neutrals": 0,
        "reversal_pressure_overrides": 0,
        "neutral_promotions": 0,
    }

    for row in history_rows:
        stats["rows_seen"] += 1
        if not isinstance(row, dict):
            continue
        new_row = dict(row)
        models = row.get("models") if isinstance(row.get("models"), list) else None
        if models is None:
            replay_rows.append(new_row)
            continue
        stats["rows_with_models"] += 1
        new_models = list(models)

        if any(str(m.get("model_id") or "") == MODEL_V2_2 for m in models if isinstance(m, dict)):
            stats["rows_with_existing_v2_2"] += 1
            new_row["models"] = new_models
            replay_rows.append(new_row)
            continue

        base = next((m for m in models if isinstance(m, dict) and str(m.get("model_id") or "") == MODEL_V2_1), None)
        if base is None:
            stats["rows_skipped_missing_base"] += 1
            new_row["models"] = new_models
            replay_rows.append(new_row)
            continue

        stats["rows_with_v2_1"] += 1
        feats = _build_features_from_history_model(base, row)
        try:
            v2_2_out = model_v2_2.run_model(feats, v2_2_config)
            v2_2 = _build_v2_2_replay_block(base, v2_2_out if isinstance(v2_2_out, dict) else {})
            debug = v2_2.get("debug") if isinstance(v2_2.get("debug"), dict) else {}
            counts = ((debug.get("v2_2") or {}).get("override_impact") or {}) if isinstance(debug.get("v2_2"), dict) else {}
            for k in [
                "continuation_protection_blocks",
                "collapse_reversal_overrides",
                "false_snap_blocks",
                "inertia_wall_neutrals",
                "reversal_pressure_overrides",
                "neutral_promotions",
            ]:
                stats[k] += int(bool(counts.get(k, 0)))
        except Exception:
            v2_2 = dict(base)
            v2_2["model_id"] = MODEL_V2_2
            v2_2["trend"] = "Neutral"
            v2_2["reason"] = f"{base.get('reason')}|v2_2_replay=model_error"

        new_models.append(v2_2)
        log.debug("[v2_2_replay_payload] %s", _replay_payload_health(v2_2))
        new_row["models"] = new_models
        replay_rows.append(new_row)
        stats["rows_with_v2_2_appended"] += 1

    return replay_rows, stats

def write_prediction_history(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
def load_prediction_history(path: Path) -> List[Dict[str, Any]]:
    """Load prediction history JSONL lines safely."""
    if not path.exists():
        log.warning("Prediction history file not found: %s", path)
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                log.warning("Skipping malformed JSONL line %s in %s", line_no, path)
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)

    return rows


def load_round_record(path: Path) -> List[Dict[str, Any]]:
    """Load round record payload as list of row dicts."""
    if not path.exists():
        log.warning("Round record file not found: %s", path)
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Failed parsing round record JSON: %s", path)
        return []

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def flatten_model_rows(history_rows: Iterable[Dict[str, Any]], model_id: str) -> List[Dict[str, Any]]:
    """Flatten selected model rows from history payload."""
    flat: List[Dict[str, Any]] = []
    for history_row in history_rows:
        models = history_row.get("models")
        if not isinstance(models, list):
            continue

        for model_row in models:
            if not isinstance(model_row, dict):
                continue
            if str(model_row.get("model_id") or "") != model_id:
                continue
            flat.append(
                {
                    "history": history_row,
                    "model": model_row,
                }
            )
    return flat

def join_truth(
    flat_rows: Iterable[Dict[str, Any]],
    round_rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Join prediction rows with round-record truth data using:

        history.next_epoch  ->  round_record.nextEpoch

    Returns a list of flattened rows ready for derived feature processing.
    """

    # -------------------------------------------------
    # Build lookup table: nextEpoch -> round_row
    # -------------------------------------------------
    round_lookup: Dict[int, Dict[str, Any]] = {}

    for round_row in round_rows:
        if not isinstance(round_row, dict):
            continue

        next_epoch = _as_int(round_row.get("nextEpoch"))
        if next_epoch is None:
            continue

        round_lookup[next_epoch] = round_row

    output: List[Dict[str, Any]] = []

    # -------------------------------------------------
    # Iterate flattened prediction rows
    # -------------------------------------------------
    for block in flat_rows:

        if not isinstance(block, dict):
            continue

        history = block.get("history")
        model = block.get("model")

        if not isinstance(history, dict) or not isinstance(model, dict):
            continue

        debug = model.get("debug") if isinstance(model.get("debug"), dict) else {}
        signals = debug.get("signals") if isinstance(debug.get("signals"), dict) else {}
        raw = model.get("raw_features_used") if isinstance(model.get("raw_features_used"), dict) else {}
        fan_physics = debug.get("fan_physics") if isinstance(debug.get("fan_physics"), dict) else {}
        fan_geo = fan_physics.get("geometry") if isinstance(fan_physics.get("geometry"), dict) else {}
        fan_hinge = fan_physics.get("hinge") if isinstance(fan_physics.get("hinge"), dict) else {}
        fan_snap = fan_physics.get("snap") if isinstance(fan_physics.get("snap"), dict) else {}
        fan_phase = fan_physics.get("phase") if isinstance(fan_physics.get("phase"), dict) else {}
        fan_energy = fan_physics.get("energy") if isinstance(fan_physics.get("energy"), dict) else {}
        context = debug.get("context") if isinstance(debug.get("context"), dict) else {}
        v2_2_block = debug.get("v2_2") if isinstance(debug.get("v2_2"), dict) else {}
        sigma_features = _extract_sigma_level_features(model, debug, raw)

        next_epoch = _as_int(history.get("next_epoch"))

        matched_round: Optional[Dict[str, Any]] = None
        if next_epoch is not None:
            matched_round = round_lookup.get(next_epoch)

        # -------------------------------------------------
        # Extract round truth values
        # -------------------------------------------------
        price_difference = None
        start_price = None
        end_price = None
        round_ts = None
        round_next_epoch_time = None

        if matched_round:
            price_difference = _as_float(matched_round.get("priceDifference"))
            start_price = _as_float(matched_round.get("startPrice"))
            end_price = _as_float(matched_round.get("endPrice"))
            round_ts = matched_round.get("current_timestamp")
            round_next_epoch_time = matched_round.get("nextEpochTime")

        truth = _truth_from_price_diff(price_difference)
        prediction = _norm_direction(model.get("trend"))

        # -------------------------------------------------
        # Build flattened row
        # -------------------------------------------------
        row: Dict[str, Any] = {
            "model_id": str(model.get("model_id") or ""),
            "epoch": _as_int(history.get("epoch")),
            "next_epoch": next_epoch,
            "timestamp": history.get("timestamp"),

            "prediction": prediction,
            "confidence": _as_float(model.get("confidence")),
            "score": _as_float(model.get("score")),
            "reason": model.get("reason"),

            "truth": truth,
            "start_price": start_price,
            "end_price": end_price,
            "price_difference": price_difference,
            "epoch_start_price": start_price,
            "epoch_end_price": end_price,
            "epoch_price_diff": price_difference,
            "round_current_timestamp": round_ts,
            "round_next_epoch_time": round_next_epoch_time,

            "regime": _normalize_regime(debug.get("regime")),
            "run_direction": str(debug.get("run_direction") or ""),

            # Signals
            "fan_width": _as_float(signals.get("fan_width")),
            "alignment": _as_float(signals.get("alignment")),
            "compression_velocity": _as_float(signals.get("compression_velocity")),
            "flip_score": _as_float(signals.get("flip_score")),
            "slope_sum": _as_float(signals.get("slope_sum")),
            "slope_magnitude": _as_float(signals.get("slope_magnitude")),
            "ratio_8_23_to_23_53": _as_float(signals.get("ratio_8_23_to_23_53")),

            # Raw features
            "s23": _as_float(raw.get("s23")),
            "s53": _as_float(raw.get("s53")),
            "fan_width_raw": _as_float(raw.get("fan_width")),
            "alignment_raw": _as_float(raw.get("alignment")),
            "compression_velocity_raw": _as_float(raw.get("compression_velocity")),
            "flip_score_raw": _as_float(raw.get("flip_score")),
            "phase_disagreement": bool(fan_phase.get("phase_disagreement")) if fan_phase else None,
            "outer_fan_width": _as_float(fan_geo.get("outer_fan_width")),
            "outer_order_signature": fan_geo.get("outer_order_signature"),
            "slow_retention": _as_float(fan_energy.get("slow_retention")),
            "reversal_pressure": _as_float(fan_snap.get("reversal_pressure")) if _as_float(fan_snap.get("reversal_pressure")) is not None else _as_float(signals.get("reversal_pressure")),
            "tail_anchor_type": context.get("tail_anchor_type"),
            "extrema_pair": context.get("extrema_pair"),
            "v2_2_rule": v2_2_block.get("rule"),
        }
        row.update(sigma_features)

        # -------------------------------------------------
        # Evaluation flags
        # -------------------------------------------------
        row["directional_called"] = row["prediction"] in {"Bull", "Bear"}
        row["directional_correct"] = bool(
            row["directional_called"] and row["prediction"] == row["truth"]
        )

        row["correct"] = bool(
            row["prediction"] == row["truth"] and row["truth"] != "Unknown"
        )

        row["bull_called"] = row["prediction"] == "Bull"
        row["bear_called"] = row["prediction"] == "Bear"
        row["neutral_called"] = row["prediction"] == "Neutral"

        # -------------------------------------------------
        # Error taxonomy flags
        # -------------------------------------------------
        row["bull_called_bear_truth"] = bool(row["bull_called"] and row["truth"] == "Bear")
        row["bear_called_bull_truth"] = bool(row["bear_called"] and row["truth"] == "Bull")
        row["neutral_called_bull_truth"] = bool(row["neutral_called"] and row["truth"] == "Bull")
        row["neutral_called_bear_truth"] = bool(row["neutral_called"] and row["truth"] == "Bear")
        row["bull_called_neutral_truth"] = bool(row["bull_called"] and row["truth"] == "Neutral")
        row["bear_called_neutral_truth"] = bool(row["bear_called"] and row["truth"] == "Neutral")

        output.append(row)

    return output


def add_derived_columns(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append derived diagnostics columns used by v2.x/v3.0 analysis."""
    prepared = [dict(r) for r in rows]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in prepared:
        grouped.setdefault(str(row.get("model_id") or MODEL_DEFAULT), []).append(row)
    derived_rows: List[Dict[str, Any]] = []
    for _, model_rows in grouped.items():
        sorted_rows = sorted(model_rows, key=_resolve_sort_key)
        prev: Optional[Dict[str, Any]] = None
        rolling_phase_strength_max = EPSILON
        velocity_history: List[float] = []
        price_diff_history: List[float] = []
        fan_energy_history: List[float] = []

        for row in sorted_rows:
            s23 = _as_float(row.get("s23"))
            s53 = _as_float(row.get("s53"))
            slope_sum = _as_float(row.get("slope_sum")) or 0.0
            fan_width = _as_float(row.get("fan_width")) or 0.0
            alignment = _as_float(row.get("alignment"))
            compression_velocity = _as_float(row.get("compression_velocity"))
            flip_score = _as_float(row.get("flip_score"))

            if _as_float(row.get("slope_g23")) is None and s23 is not None:
                row["slope_g23"] = s23
            if _as_float(row.get("slope_g53")) is None and s53 is not None:
                row["slope_g53"] = s53

            slope_sign_s23 = _sign(s23)
            slope_sign_s53 = _sign(s53)
            slope_disagreement = slope_sign_s23 != slope_sign_s53
            slope_gap = abs((s23 or 0.0) - (s53 or 0.0))
            abs_slope_sum = abs(slope_sum)
            abs_compression_velocity = abs(compression_velocity or 0.0)
            abs_flip_score = abs(flip_score or 0.0)
            fan_to_slope_ratio = fan_width / max(abs_slope_sum, EPSILON)

            collapse_flag = bool(compression_velocity is not None and alignment is not None and compression_velocity < COLLAPSE_COMPRESSION_THRESHOLD and alignment < WEAK_ALIGNMENT_THRESHOLD)
            weak_bull_exhaustion_flag = bool(slope_sum > 0 and compression_velocity is not None and alignment is not None and flip_score is not None and compression_velocity < COLLAPSE_COMPRESSION_THRESHOLD and alignment < WEAK_ALIGNMENT_THRESHOLD and flip_score > FLIP_SCORE_ALERT_THRESHOLD)
            weak_bear_exhaustion_flag = bool(slope_sum < 0 and compression_velocity is not None and alignment is not None and flip_score is not None and compression_velocity < COLLAPSE_COMPRESSION_THRESHOLD and alignment < WEAK_ALIGNMENT_THRESHOLD and flip_score < -FLIP_SCORE_ALERT_THRESHOLD)
            possible_flip_zone = bool(collapse_flag or weak_bull_exhaustion_flag or weak_bear_exhaustion_flag or ((flip_score or 0.0) > FLIP_SCORE_ALERT_THRESHOLD and (alignment or 0.0) < WEAK_ALIGNMENT_THRESHOLD))
            possible_run_zone = bool((alignment or 0.0) >= RUN_ALIGNMENT_MIN and abs((_as_float(row.get("compression_velocity")) or 0.0)) <= NOISE_SLOPE_MAX and (_as_float(row.get("ratio_8_23_to_23_53")) or 0.0) >= FLIP_ZONE_RATIO_THRESHOLD)
            possible_transition_zone = bool((not possible_run_zone) and possible_flip_zone)

            geometry = compute_geometry_features(row, prev)
            row.update(geometry)
            energy = compute_energy_features(row, prev)
            row.update(energy)
            hinge = compute_hinge_features(row, prev)
            row.update(hinge)
            snap = compute_snap_features(row, prev)
            row.update(snap)

            if row.get("fan_energy_total") is not None:
                fan_energy_history.append(float(row["fan_energy_total"]))
                rolling_phase_strength_max = max(rolling_phase_strength_max, float(row["fan_energy_total"]))
            phase = compute_phase_features(row, prev, rolling_phase_strength_max)
            velocity_now = _as_float(row.get("fan_width_velocity"))
            if velocity_now is not None:
                velocity_history.append(velocity_now)
            velocity_window = velocity_history[-3:]

            price_diff = _as_float(row.get("price_difference"))
            if price_diff is not None:
                price_diff_history.append(price_diff)

            gaussian_wave = compute_gaussian_wave(velocity_window)
            wave_strength = compute_wave_strength(velocity_window)
            slow_retention = compute_slow_retention(row)
            outer_fan_width = compute_outer_fan_width(row)
            phase_disagreement = compute_phase_disagreement(row)
            row["gaussian_wave"] = gaussian_wave
            row["wave_strength"] = wave_strength
            row["slow_retention"] = slow_retention
            row["outer_fan_width"] = outer_fan_width
            row["phase_disagreement"] = phase_disagreement
            row["inertia_wall_score"] = compute_inertia_wall_score(row)
            # Transition/collapse and polarity research features (offline exporter only).
            gvals = [_as_float(row.get(k)) for k in ["g8", "g23", "g38", "g53", "g68", "g83"]]
            fan_polarity_normal = int(all(gvals[i] is not None and gvals[i] > gvals[i + 1] for i in range(len(gvals) - 1)))
            fan_polarity_inversion = int(all(gvals[i] is not None and gvals[i] < gvals[i + 1] for i in range(len(gvals) - 1)))
            hinge_torsion = _as_float(row.get("hinge_torsion")) or 0.0
            hinge_torsion_spike = int(abs(hinge_torsion) > HINGE_TORSION_SPIKE_THRESHOLD)
            transition_score_base = int((1 if phase_disagreement == 1 else 0) + hinge_torsion_spike + fan_polarity_inversion)

            outer_prev = _as_float((prev or {}).get("outer_fan_width"))
            row["outer_fan_width_prev"] = outer_prev
            outer_shrinking = (outer_fan_width is not None and outer_prev is not None and outer_fan_width < outer_prev)
            fwv = _as_float(row.get("fan_width_velocity")) or 0.0
            row["fan_collapse_v2"] = int(fwv < FAN_COLLAPSE_V2_VELOCITY_THRESHOLD and outer_shrinking and phase_disagreement == 1 and abs(hinge_torsion) > 0.6)
            row["fan_collapse_v2_strict"] = int(fwv < FAN_COLLAPSE_V2_STRICT_VELOCITY_THRESHOLD and outer_shrinking and phase_disagreement == 1 and abs(hinge_torsion) > 0.6)
            row["compression_trap"] = int(fwv < COMPRESSION_TRAP_VELOCITY_THRESHOLD and outer_shrinking and phase_disagreement == 0 and abs(hinge_torsion) <= 0.6 and (gaussian_wave or 0) == 0)
            row["compression_trap_strict"] = int(fwv < COMPRESSION_TRAP_STRICT_VELOCITY_THRESHOLD and outer_shrinking and phase_disagreement == 0 and abs(hinge_torsion) <= 0.6 and (gaussian_wave or 0) == 0)
            row["hinge_torsion_spike"] = hinge_torsion_spike
            row["transition_score_base"] = transition_score_base
            row["fan_polarity_inversion"] = fan_polarity_inversion
            row["fan_polarity_normal"] = fan_polarity_normal

            v2 = velocity_window[-3] if len(velocity_window) >= 3 else None
            v1 = velocity_window[-2] if len(velocity_window) >= 2 else None
            v0 = velocity_window[-1] if len(velocity_window) >= 1 else None
            row["fan_velocity_sign_t2"] = _sign(v2)
            row["fan_velocity_sign_t1"] = _sign(v1)
            row["fan_velocity_sign_t0"] = _sign(v0)
            row["fan_velocity_pattern_3"] = f"{_sign_to_char(v2)}{_sign_to_char(v1)}{_sign_to_char(v0)}"
            row["fan_velocity_oscillation"] = int(row["fan_velocity_pattern_3"] in {"+-+", "-+-"})
            row["fan_velocity_oscillation_strength"] = abs(v2 or 0.0) + abs(v1 or 0.0) + abs(v0 or 0.0)

            row["epoch_start_price"] = _as_float(row.get("start_price"))
            row["epoch_end_price"] = _as_float(row.get("end_price"))
            row["epoch_price_diff"] = _as_float(row.get("price_difference"))
            sp = row["epoch_start_price"] or 0.0
            pdiff = row["epoch_price_diff"] or 0.0
            row["epoch_return"] = pdiff / max(abs(sp), EPSILON)
            row["epoch_volatility"] = abs(pdiff)
            p2 = price_diff_history[-3] if len(price_diff_history) >= 3 else None
            p1 = price_diff_history[-2] if len(price_diff_history) >= 2 else None
            p0 = price_diff_history[-1] if len(price_diff_history) >= 1 else None
            row["price_diff_t2"] = p2
            row["price_diff_t1"] = p1
            row["price_diff_t0"] = p0
            row["momentum_3epoch"] = (p2 or 0.0) + (p1 or 0.0) + (p0 or 0.0)
            row["price_diff_3epoch_abs_sum"] = abs(p2 or 0.0) + abs(p1 or 0.0) + abs(p0 or 0.0)
            row["price_diff_sign_pattern"] = f"{_sign_to_char(p2)}{_sign_to_char(p1)}{_sign_to_char(p0)}"
            row["direction_flip"] = int(_sign(p0) != _sign(p1))
            row["price_diff_decay_ratio"] = abs(p0 or 0.0) / max(abs(p2 or 0.0), EPSILON)
            row["price_diff_vol_squeeze"] = int(abs(p2 or 0.0) > abs(p1 or 0.0) > abs(p0 or 0.0))

            e0 = fan_energy_history[-1] if len(fan_energy_history) >= 1 else None
            e1 = fan_energy_history[-2] if len(fan_energy_history) >= 2 else None
            e2 = fan_energy_history[-3] if len(fan_energy_history) >= 3 else None
            e3 = fan_energy_history[-4] if len(fan_energy_history) >= 4 else None
            row["fan_energy_t0"] = e0
            row["fan_energy_t1"] = e1
            row["fan_energy_t2"] = e2
            row["fan_energy_t3"] = e3
            row["fan_energy_release_signal"] = int((e3 is not None and e2 is not None and e1 is not None and e0 is not None) and (e3 > e2 > e1 and e0 > e1))
            if e2 is not None and e1 is not None and e0 is not None and e2 > e1 > e0:
                row["fan_energy_cycle_stage"] = "decay"
            elif e2 is not None and e1 is not None and e0 is not None and e2 < e1 < e0:
                row["fan_energy_cycle_stage"] = "build"
            elif row["fan_energy_release_signal"] == 1:
                row["fan_energy_cycle_stage"] = "release"
            else:
                row["fan_energy_cycle_stage"] = "other"
            fan_width_acc = _as_float(row.get("fan_width_acceleration"))
            hinge_velocity = _as_float(row.get("hinge_velocity"))
            snap_velocity = _as_float(row.get("snap_velocity"))
            reversal_pressure_terms = []
            if fan_width_acc is not None:
                reversal_pressure_terms.append(-fan_width_acc)
            if hinge_velocity is not None:
                reversal_pressure_terms.append(abs(hinge_velocity))
            if snap_velocity is not None:
                reversal_pressure_terms.append(snap_velocity)
            reversal_pressure = sum(reversal_pressure_terms) if reversal_pressure_terms else _as_float(row.get("reversal_pressure"))

            truth = row.get("truth") if row.get("truth") in {"Bull", "Bear", "Neutral"} else "Unknown"
            pred = row.get("prediction") if row.get("prediction") in {"Bull", "Bear", "Neutral"} else "Unknown"

            directional_error_type = "none"
            if pred == "Bull" and truth == "Bear":
                directional_error_type = "bull_called_bear_truth"
            elif pred == "Bear" and truth == "Bull":
                directional_error_type = "bear_called_bull_truth"
            elif pred == "Neutral" and truth == "Bull":
                directional_error_type = "neutral_called_bull_truth"
            elif pred == "Neutral" and truth == "Bear":
                directional_error_type = "neutral_called_bear_truth"
            elif pred == "Bull" and truth == "Bull":
                directional_error_type = "correct_bull"
            elif pred == "Bear" and truth == "Bear":
                directional_error_type = "correct_bear"
            elif pred == "Neutral" and truth == "Neutral":
                directional_error_type = "correct_neutral"

            row.update({
                "slope_sign_s23": slope_sign_s23,
                "slope_sign_s53": slope_sign_s53,
                "slope_disagreement": slope_disagreement,
                "slope_gap": slope_gap,
                "abs_slope_sum": abs_slope_sum,
                "abs_compression_velocity": abs_compression_velocity,
                "abs_flip_score": abs_flip_score,
                "fan_to_slope_ratio": fan_to_slope_ratio,
                "collapse_flag": collapse_flag,
                "weak_bull_exhaustion_flag": weak_bull_exhaustion_flag,
                "weak_bear_exhaustion_flag": weak_bear_exhaustion_flag,
                "possible_flip_zone": possible_flip_zone,
                "possible_run_zone": possible_run_zone,
                "possible_transition_zone": possible_transition_zone,
                "phase_alignment": phase["phase_alignment"],
                "phase_strength": phase["phase_strength"],
                "phase_velocity": phase["phase_velocity"],
                "fan_phase_score": phase["fan_phase_score"],
                "reversal_pressure": reversal_pressure,
                "truth_direction_group": truth,
                "prediction_direction_group": pred,
                "directional_error_type": directional_error_type,
            })
            row["prev_g83"] = _as_float((prev or {}).get("g83"))
            row["phase_input_count"] = phase["phase_input_count"]
            prev = row
            derived_rows.append(row)

    return sorted(derived_rows, key=lambda r: (str(r.get("model_id") or MODEL_DEFAULT),) + _resolve_sort_key(r))


def build_feature_population_audit(rows: Sequence[Dict[str, Any]]) -> Tuple[
    List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Build non-null/non-zero population diagnostics for sigma and advanced feature families."""
    total_rows = len(rows)
    metrics = [
        "g8", "g23", "g38", "g53", "g68", "g83",
        "slope_g8", "slope_g23", "slope_g38", "slope_g53", "slope_g68", "slope_g83",
        "fan_width", "fan_order_score", "g83_curvature", "fan_energy_total", "fan_energy_ratio",
        "hinge_gap", "hinge_torsion", "snap_score", "phase_alignment", "fan_phase_score",
        "gaussian_wave", "wave_strength", "slow_retention", "outer_fan_width", "phase_disagreement",
        "inertia_wall_score",
    ]
    audit_rows: List[Dict[str, Any]] = []
    summary_counts: Dict[str, Any] = {"total_rows": total_rows}

    for feature in metrics:
        values = [_as_float(r.get(feature)) for r in rows]
        non_null_count = sum(1 for v in values if v is not None)
        non_zero_count = sum(1 for v in values if v is not None and abs(v) > 0.0)
        audit_rows.append(
    {
        "feature_name": feature,
        "non_null_count": non_null_count,
        "non_zero_count": non_zero_count,
        "total_rows": total_rows,
        "population_rate": _safe_ratio(non_null_count, total_rows),
        "non_zero_rate": _safe_ratio(non_zero_count, total_rows),
    }
)
        summary_counts[f"rows_with_{feature}"] = non_null_count

    summary_counts["rows_with_nonzero_fan_order_score"] = sum(
                1 for r in rows if (_as_float(r.get("fan_order_score")) or 0.0) != 0.0)
    summary_counts["rows_with_nonzero_hinge_torsion"] = sum(
                1 for r in rows if (_as_float(r.get("hinge_torsion")) or 0.0) != 0.0)
    summary_counts["rows_with_nonzero_snap_score"] = sum(
                1 for r in rows if (_as_float(r.get("snap_score")) or 0.0) != 0.0)
    summary_counts["rows_with_nonzero_phase_alignment"] = sum(
                1 for r in rows if (_as_float(r.get("phase_alignment")) or 0.0) != 0.0)
    summary_counts["rows_with_valid_phase_inputs_ge_4"] = sum(
                1 for r in rows if (_as_int(r.get("phase_input_count")) or 0) >= 4)

    warnings: List[str] = []
    for feat, hint in [
        ("fan_order_score", "check sigma extraction"),
        ("hinge_torsion", "check slope_g23/slope_g38 extraction"),
        ("snap_score", "check slope_g8/slope_g23/slope_g38 extraction"),
        ("phase_alignment", "check multi-sigma slope extraction"),
    ]:
        audit = next((r for r in audit_rows if r["feature_name"] == feat), None)
        if not audit:
            continue
        if audit["non_null_count"] <= max(1, int(total_rows * 0.05)) or audit["non_zero_count"] <= max(1,
                                                                                                               int(total_rows * 0.05)):
            warnings.append(f"WARNING: {feat} population too low; {hint}")

    return audit_rows, summary_counts, warnings

def write_feature_population_audit(path: Path, audit_rows: Sequence[Dict[str, Any]]) -> None:
    """Persist feature population audit CSV."""
    _write_csv(path, audit_rows,
                ["feature_name", "non_null_count", "non_zero_count", "total_rows", "population_rate",
                "non_zero_rate"])

def write_feature_source_map(path: Path, model_audits: Dict[str, Dict[str, Any]]) -> None:
    """Document sigma extraction and observed feature population coverage per model."""
    lines = [
        "gaussian_fan_feature_source_map",
        "",
        "Sigma extraction sources:",
        "- top-level model fields (g8..g83, slope_g8..slope_g83)",
        "- model.debug.signals aliases (gauss/sigma/slope variants)",
        "- model.raw_features_used aliases",
        "- nested containers: gaussian_levels/sigma_levels and gaussian_slopes/sigma_slopes",
        "",
    ]
    for model_id, counts in model_audits.items():
        lines.append(f"[{model_id}]")
        for key in ["g8", "g23", "g38", "g53", "g68", "g83", "slope_g8", "slope_g23", "slope_g38", "slope_g53",
                    "slope_g68", "slope_g83", "fan_order_score", "hinge_torsion", "snap_score",
                    "phase_alignment"]:
            lines.append(f"- rows_with_{key}: {int(counts.get(f'rows_with_{key}', 0))}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _build_group_stats(rows: Sequence[Dict[str, Any]], group_name: str, group_value: str) -> Dict[str, Any]:
    directional_rows = [r for r in rows if r.get("directional_called")]
    pred_bull = [r for r in rows if r.get("prediction") == "Bull"]
    pred_bear = [r for r in rows if r.get("prediction") == "Bear"]
    pred_neutral = [r for r in rows if r.get("prediction") == "Neutral"]

    correct_bull = [r for r in rows if r.get("prediction") == "Bull" and r.get("truth") == "Bull"]
    correct_bear = [r for r in rows if r.get("prediction") == "Bear" and r.get("truth") == "Bear"]

    return {
        "group_name": group_name,
        "group_value": group_value,
        "row_count": len(rows),
        "directional_called": len(directional_rows),
        "directional_correct": sum(1 for r in directional_rows if r.get("directional_correct")),
        "directional_accuracy": _safe_ratio(
            sum(1 for r in directional_rows if r.get("directional_correct")),
            len(directional_rows),
        ),
        "coverage_within_export_set": 0.0,
        "bull_precision": _safe_ratio(len(correct_bull), len(pred_bull)),
        "bear_precision": _safe_ratio(len(correct_bear), len(pred_bear)),
        "neutral_rate": _safe_ratio(len(pred_neutral), len(rows)),
        "mean_s23": _safe_mean(rows, "s23"),
        "mean_s53": _safe_mean(rows, "s53"),
        "mean_slope_sum": _safe_mean(rows, "slope_sum"),
        "mean_slope_magnitude": _safe_mean(rows, "slope_magnitude"),
        "mean_fan_width": _safe_mean(rows, "fan_width"),
        "mean_alignment": _safe_mean(rows, "alignment"),
        "mean_compression_velocity": _safe_mean(rows, "compression_velocity"),
        "mean_flip_score": _safe_mean(rows, "flip_score"),
        "mean_ratio_8_23_to_23_53": _safe_mean(rows, "ratio_8_23_to_23_53"),
        "mean_slope_gap": _safe_mean(rows, "slope_gap"),
        "mean_fan_width_velocity": _safe_mean(rows, "fan_width_velocity"),
        "mean_fan_width_acceleration": _safe_mean(rows, "fan_width_acceleration"),
        "mean_fan_order_score": _safe_mean(rows, "fan_order_score"),
        "count_fan_order_score": sum(1 for r in rows if _as_float(r.get("fan_order_score")) is not None),
        "mean_g83_curvature": _safe_mean(rows, "g83_curvature"),
        "mean_fan_energy_total": _safe_mean(rows, "fan_energy_total"),
        "mean_fan_energy_ratio": _safe_mean(rows, "fan_energy_ratio"),
        "mean_fan_energy_velocity": _safe_mean(rows, "fan_energy_velocity"),
        "mean_fan_energy_acceleration": _safe_mean(rows, "fan_energy_acceleration"),
        "fan_sign_conflict_rate": _safe_ratio(sum(1 for r in rows if r.get("fan_sign_conflict")), len(rows)),
        "mean_fan_energy_instability": _safe_mean(rows, "fan_energy_instability"),
        "mean_hinge_gap": _safe_mean(rows, "hinge_gap"),
        "mean_hinge_velocity": _safe_mean(rows, "hinge_velocity"),
        "hinge_conflict_rate": _safe_ratio(sum(1 for r in rows if r.get("hinge_conflict")), len(rows)),
        "mean_hinge_torsion": _safe_mean(rows, "hinge_torsion"),
        "count_hinge_torsion": sum(1 for r in rows if _as_float(r.get("hinge_torsion")) is not None),
        "mean_snap_divergence": _safe_mean(rows, "snap_divergence"),
        "mean_snap_velocity": _safe_mean(rows, "snap_velocity"),
        "mean_snap_score": _safe_mean(rows, "snap_score"),
        "count_snap_score": sum(1 for r in rows if _as_float(r.get("snap_score")) is not None),
        "mean_phase_alignment": _safe_mean(rows, "phase_alignment"),
        "count_phase_alignment": sum(1 for r in rows if _as_float(r.get("phase_alignment")) is not None),
        "mean_phase_strength": _safe_mean(rows, "phase_strength"),
        "mean_phase_velocity": _safe_mean(rows, "phase_velocity"),
        "mean_fan_phase_score": _safe_mean(rows, "fan_phase_score"),
        "mean_gaussian_wave": _safe_mean(rows, "gaussian_wave"),
        "count_gaussian_wave": sum(1 for r in rows if _as_float(r.get("gaussian_wave")) is not None),
        "mean_wave_strength": _safe_mean(rows, "wave_strength"),
        "count_wave_strength": sum(1 for r in rows if _as_float(r.get("wave_strength")) is not None),
        "mean_slow_retention": _safe_mean(rows, "slow_retention"),
        "count_slow_retention": sum(1 for r in rows if _as_float(r.get("slow_retention")) is not None),
        "mean_outer_fan_width": _safe_mean(rows, "outer_fan_width"),
        "count_outer_fan_width": sum(1 for r in rows if _as_float(r.get("outer_fan_width")) is not None),
        "mean_phase_disagreement": _safe_mean(rows, "phase_disagreement"),
        "count_phase_disagreement": sum(1 for r in rows if _as_float(r.get("phase_disagreement")) is not None),
        "mean_inertia_wall_score": _safe_mean(rows, "inertia_wall_score"),
        "count_inertia_wall_score": sum(1 for r in rows if _as_float(r.get("inertia_wall_score")) is not None),
        "collapse_flag_rate": _safe_ratio(sum(1 for r in rows if r.get("collapse_flag")), len(rows)),
        "weak_bull_exhaustion_flag_rate": _safe_ratio(
            sum(1 for r in rows if r.get("weak_bull_exhaustion_flag")), len(rows)
        ),
        "weak_bear_exhaustion_flag_rate": _safe_ratio(
            sum(1 for r in rows if r.get("weak_bear_exhaustion_flag")), len(rows)
        ),
    }

def _group_rows(rows: Sequence[Dict[str, Any]], key_name: str, values: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: List[Dict[str, Any]] = []
    for val in values:
        subset = [r for r in rows if str(r.get(key_name)) == val]
        grouped.append(_build_group_stats(subset, key_name, val))
    total = max(1, len(rows))
    for record in grouped:
        record["coverage_within_export_set"] = _safe_ratio(record["row_count"], total)
    return grouped

def build_regime_breakdown(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _group_rows(rows, "regime", ["RUN", "REVERSAL", "NOISE", "UNKNOWN"])

def build_error_breakdown(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    error_values = [
        "bull_called_bear_truth",
        "bear_called_bull_truth",
        "neutral_called_bull_truth",
        "neutral_called_bear_truth",
        "correct_bull",
        "correct_bear",
        "correct_neutral",
    ]
    out = _group_rows(rows, "directional_error_type", error_values)

    wrong_bull_bear = [r for r in rows if r.get("directional_error_type") == "bull_called_bear_truth"]
    correct_bull = [r for r in rows if r.get("directional_error_type") == "correct_bull"]
    for label, subset in [
        ("v2_1_wrong_bull_to_bear", wrong_bull_bear),
        ("v2_1_correct_bull", correct_bull),
    ]:
        stats = _build_group_stats(subset, "v2_1_diagnostic_group", label)
        stats["slope_disagreement_rate"] = _safe_ratio(
            sum(1 for r in subset if r.get("slope_disagreement")),
            len(subset),
        )
        out.append(stats)
    return out

def build_run_flip_breakdown(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("possible_run_zone", [r for r in rows if r.get("possible_run_zone")]),
        ("possible_flip_zone", [r for r in rows if r.get("possible_flip_zone")]),
        ("possible_transition_zone", [r for r in rows if r.get("possible_transition_zone")]),
    ]
    output = []
    total = max(1, len(rows))
    for name, subset in groups:
        stats = _build_group_stats(subset, "proxy_zone", name)
        stats["coverage_within_export_set"] = _safe_ratio(stats["row_count"], total)
        output.append(stats)
    return output

def build_v2_2_feature_preview(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact v2_2/v2_3 research aggregates for JSON summaries and terminal previews."""
    return {
        "geometry": {
            "mean_fan_width": _safe_mean(rows, "fan_width"),
            "mean_fan_width_velocity": _safe_mean(rows, "fan_width_velocity"),
            "mean_fan_width_acceleration": _safe_mean(rows, "fan_width_acceleration"),
            "mean_fan_order_score": _safe_mean(rows, "fan_order_score"),
            "mean_g83_curvature": _safe_mean(rows, "g83_curvature"),
        },
        "energy": {
            "mean_fan_energy_total": _safe_mean(rows, "fan_energy_total"),
            "mean_fan_energy_ratio": _safe_mean(rows, "fan_energy_ratio"),
            "mean_fan_energy_velocity": _safe_mean(rows, "fan_energy_velocity"),
            "mean_fan_energy_acceleration": _safe_mean(rows, "fan_energy_acceleration"),
            "fan_sign_conflict_rate": _safe_ratio(sum(1 for r in rows if r.get("fan_sign_conflict")), len(rows)),
            "mean_fan_energy_instability": _safe_mean(rows, "fan_energy_instability"),
        },
        "hinge": {
            "mean_hinge_gap": _safe_mean(rows, "hinge_gap"),
            "mean_hinge_velocity": _safe_mean(rows, "hinge_velocity"),
            "hinge_conflict_rate": _safe_ratio(sum(1 for r in rows if r.get("hinge_conflict")), len(rows)),
            "mean_hinge_torsion": _safe_mean(rows, "hinge_torsion"),
        },
        "snap": {
            "mean_snap_divergence": _safe_mean(rows, "snap_divergence"),
            "mean_snap_velocity": _safe_mean(rows, "snap_velocity"),
            "mean_snap_score": _safe_mean(rows, "snap_score"),
        },
        "phase": {
            "mean_phase_alignment": _safe_mean(rows, "phase_alignment"),
            "mean_phase_strength": _safe_mean(rows, "phase_strength"),
            "mean_phase_velocity": _safe_mean(rows, "phase_velocity"),
            "mean_fan_phase_score": _safe_mean(rows, "fan_phase_score"),
        },
        "wave_inertia": {
            "gaussian_wave_rate": _safe_mean(rows, "gaussian_wave"),
            "mean_wave_strength": _safe_mean(rows, "wave_strength"),
            "mean_slow_retention": _safe_mean(rows, "slow_retention"),
            "phase_disagreement_rate": _safe_mean(rows, "phase_disagreement"),
            "mean_outer_fan_width": _safe_mean(rows, "outer_fan_width"),
            "mean_inertia_wall_score": _safe_mean(rows, "inertia_wall_score"),
        },
    }


def build_v2_2_feature_candidates_v2_1(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = [
        "correct_bull",
        "correct_bear",
        "bull_called_bear_truth",
        "bear_called_bull_truth",
        "neutral_called_bull_truth",
        "neutral_called_bear_truth",
    ]
    return [
        _build_group_stats([r for r in rows if r.get("directional_error_type") == label], "directional_error_type", label)
        for label in groups
    ]

def _max_streak(rows: Sequence[Dict[str, Any]], target: str) -> int:
    streak = 0
    best = 0
    for row in rows:
        if not row.get("directional_called"):
            streak = 0
            continue
        if target == "win" and row.get("directional_correct"):
            streak += 1
        elif target == "loss" and not row.get("directional_correct"):
            streak += 1
        else:
            streak = 0
        best = max(best, streak)
    return best

def build_main_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_rows = len(rows)
    joined_truth_rows = sum(1 for r in rows if r.get("truth") in {"Bull", "Bear", "Neutral"})
    directional_called = sum(1 for r in rows if r.get("directional_called"))
    neutral_called = sum(1 for r in rows if r.get("neutral_called"))
    directional_correct = sum(1 for r in rows if r.get("directional_correct"))

    bull_pred_bull_truth = sum(1 for r in rows if r.get("prediction") == "Bull" and r.get("truth") == "Bull")
    bull_pred_bear_truth = sum(1 for r in rows if r.get("prediction") == "Bull" and r.get("truth") == "Bear")
    bear_pred_bull_truth = sum(1 for r in rows if r.get("prediction") == "Bear" and r.get("truth") == "Bull")
    bear_pred_bear_truth = sum(1 for r in rows if r.get("prediction") == "Bear" and r.get("truth") == "Bear")
    neutral_pred_bull_truth = sum(1 for r in rows if r.get("prediction") == "Neutral" and r.get("truth") == "Bull")
    neutral_pred_bear_truth = sum(1 for r in rows if r.get("prediction") == "Neutral" and r.get("truth") == "Bear")

    pred_bull_count = sum(1 for r in rows if r.get("prediction") == "Bull")
    pred_bear_count = sum(1 for r in rows if r.get("prediction") == "Bear")
    pred_neutral_count = sum(1 for r in rows if r.get("prediction") == "Neutral")

    truth_bull_count = sum(1 for r in rows if r.get("truth") == "Bull")
    truth_bear_count = sum(1 for r in rows if r.get("truth") == "Bear")
    truth_neutral_count = sum(1 for r in rows if r.get("truth") == "Neutral")

    run_rows = [r for r in rows if r.get("regime") == "RUN"]
    reversal_rows = [r for r in rows if r.get("regime") == "REVERSAL"]
    noise_rows = [r for r in rows if r.get("regime") == "NOISE"]

    return {
        "total_rows": total_rows,
        "joined_truth_rows": joined_truth_rows,
        "directional_called": directional_called,
        "neutral_called": neutral_called,
        "directional_accuracy": _safe_ratio(directional_correct, directional_called),
        "directional_coverage": _safe_ratio(directional_called, total_rows),
        "neutral_rate": _safe_ratio(neutral_called, total_rows),
        "bull_precision": _safe_ratio(bull_pred_bull_truth, pred_bull_count),
        "bear_precision": _safe_ratio(bear_pred_bear_truth, pred_bear_count),
        "truth_bull_count": truth_bull_count,
        "truth_bear_count": truth_bear_count,
        "truth_neutral_count": truth_neutral_count,
        "pred_bull_count": pred_bull_count,
        "pred_bear_count": pred_bear_count,
        "pred_neutral_count": pred_neutral_count,
        "bull_pred_bull_truth": bull_pred_bull_truth,
        "bull_pred_bear_truth": bull_pred_bear_truth,
        "bear_pred_bull_truth": bear_pred_bull_truth,
        "bear_pred_bear_truth": bear_pred_bear_truth,
        "neutral_pred_bull_truth": neutral_pred_bull_truth,
        "neutral_pred_bear_truth": neutral_pred_bear_truth,
        "run_rows": len(run_rows),
        "reversal_rows": len(reversal_rows),
        "noise_rows": len(noise_rows),
        "run_directional_accuracy": _safe_ratio(
            sum(1 for r in run_rows if r.get("directional_correct")),
            sum(1 for r in run_rows if r.get("directional_called")),
        ),
        "reversal_directional_accuracy": _safe_ratio(
            sum(1 for r in reversal_rows if r.get("directional_correct")),
            sum(1 for r in reversal_rows if r.get("directional_called")),
        ),
        "noise_directional_accuracy": _safe_ratio(
            sum(1 for r in noise_rows if r.get("directional_correct")),
            sum(1 for r in noise_rows if r.get("directional_called")),
        ),
        "possible_run_zone_rows": sum(1 for r in rows if r.get("possible_run_zone")),
        "possible_flip_zone_rows": sum(1 for r in rows if r.get("possible_flip_zone")),
        "possible_transition_zone_rows": sum(1 for r in rows if r.get("possible_transition_zone")),
        "consecutive_directional_wins_max": _max_streak(rows, "win"),
        "consecutive_directional_losses_max": _max_streak(rows, "loss"),
        "neutral_breaks_streak": True,
    }



def run_threshold_scan(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Legacy analysis-only threshold scan for Bull overrides.
    Kept for compatibility, but v2_1 work should prefer
    run_v2_1_bull_instability_scan().
    """
    compression_levels = [-0.4, -0.6, -0.8, -1.0, -1.2]
    alignment_levels = [0.15, 0.20, 0.25, 0.30, 0.40]
    flip_levels = [0.03, 0.05, 0.07, 0.10]

    results: List[Dict[str, Any]] = []
    combos = itertools.product(compression_levels, alignment_levels, flip_levels)

    baseline_summary = build_main_summary(rows)
    baseline = {
        "baseline_directional_called": baseline_summary["directional_called"],
        "baseline_directional_accuracy": baseline_summary["directional_accuracy"],
        "baseline_directional_coverage": baseline_summary["directional_coverage"],
        "baseline_bull_precision": baseline_summary["bull_precision"],
        "baseline_bear_precision": baseline_summary["bear_precision"],
    }

    for c_thr, a_thr, f_thr in combos:
        candidates = [
            r
            for r in rows
            if r.get("prediction") == "Bull"
            and (_as_float(r.get("compression_velocity")) or 0.0) <= c_thr
            and (_as_float(r.get("alignment")) or 0.0) <= a_thr
            and (_as_float(r.get("flip_score")) or 0.0) >= f_thr
        ]

        candidate_keys = {(r.get("epoch"), r.get("next_epoch"), r.get("timestamp")) for r in candidates}

        for variant in ["bull_to_neutral", "bull_to_bear"]:
            simulated: List[Dict[str, Any]] = []
            for row in rows:
                clone = dict(row)
                key = (row.get("epoch"), row.get("next_epoch"), row.get("timestamp"))
                if key in candidate_keys and row.get("prediction") == "Bull":
                    clone["prediction"] = "Neutral" if variant == "bull_to_neutral" else "Bear"
                    clone["directional_called"] = clone["prediction"] in {"Bull", "Bear"}
                    clone["directional_correct"] = bool(
                        clone["directional_called"] and clone["prediction"] == clone.get("truth")
                    )
                    clone["correct"] = bool(clone["prediction"] == clone.get("truth") and clone.get("truth") != "Unknown")
                simulated.append(clone)

            sim_summary = build_main_summary(simulated)
            results.append(
                {
                    "variant": variant,
                    "compression_threshold": c_thr,
                    "alignment_threshold": a_thr,
                    "flip_score_threshold": f_thr,
                    "changed_rows_count": len(candidates),
                    "new_directional_called": sim_summary["directional_called"],
                    **baseline,
                    "new_directional_accuracy": sim_summary["directional_accuracy"],
                    "new_directional_coverage": sim_summary["directional_coverage"],
                    "new_bull_precision": sim_summary["bull_precision"],
                    "new_bear_precision": sim_summary["bear_precision"],
                    "delta_directional_accuracy": sim_summary["directional_accuracy"] - baseline["baseline_directional_accuracy"],
                    "delta_directional_coverage": sim_summary["directional_coverage"] - baseline["baseline_directional_coverage"],
                    "delta_bull_precision": sim_summary["bull_precision"] - baseline["baseline_bull_precision"],
                    "delta_bear_precision": sim_summary["bear_precision"] - baseline["baseline_bear_precision"],
                }
            )

    return results


def run_v2_1_bull_instability_scan(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Scan analysis-only v2_1 Bull-run instability rules.

    Focus:
    - prediction == Bull
    - regime == RUN

    Candidate signal shape:
    - elevated flip_score
    - weakened alignment
    - optional compression filter
    - optional ratio filter

    Returns:
        (all_scan_rows, baseline_metrics, top_bull_to_neutral, top_bull_to_bear)
    """
    flip_thresholds = [0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25]
    alignment_thresholds = [1.00, 0.99, 0.98, 0.97, 0.95, 0.93, 0.90]
    compression_mins: List[Optional[float]] = [None, 0.0, 5.0, 10.0, 15.0]
    ratio_thresholds: List[Optional[float]] = [None, 0.0, -0.05, -0.10]

    baseline_summary = build_main_summary(rows)
    baseline = {
        "baseline_directional_called": baseline_summary["directional_called"],
        "baseline_directional_accuracy": baseline_summary["directional_accuracy"],
        "baseline_directional_coverage": baseline_summary["directional_coverage"],
        "baseline_bull_precision": baseline_summary["bull_precision"],
        "baseline_bear_precision": baseline_summary["bear_precision"],
    }

    bull_run_rows = [
        r for r in rows
        if r.get("prediction") == "Bull" and r.get("regime") == "RUN"
    ]

    results: List[Dict[str, Any]] = []

    for f_thr, a_thr, c_min, ratio_thr in itertools.product(
        flip_thresholds,
        alignment_thresholds,
        compression_mins,
        ratio_thresholds,
    ):
        candidates = [
            r
            for r in bull_run_rows
            if (_as_float(r.get("flip_score")) or 0.0) >= f_thr
            and (_as_float(r.get("alignment")) or 0.0) <= a_thr
            and (c_min is None or (_as_float(r.get("compression_velocity")) or 0.0) >= c_min)
            and (ratio_thr is None or (_as_float(r.get("ratio_8_23_to_23_53")) or 0.0) <= ratio_thr)
        ]

        candidate_keys = {
            (r.get("epoch"), r.get("next_epoch"), r.get("timestamp"))
            for r in candidates
        }

        for variant in ("bull_to_neutral", "bull_to_bear"):
            simulated: List[Dict[str, Any]] = []

            for row in rows:
                clone = dict(row)
                key = (row.get("epoch"), row.get("next_epoch"), row.get("timestamp"))

                if key in candidate_keys and row.get("prediction") == "Bull" and row.get("regime") == "RUN":
                    clone["prediction"] = "Neutral" if variant == "bull_to_neutral" else "Bear"
                    clone["directional_called"] = clone["prediction"] in {"Bull", "Bear"}
                    clone["directional_correct"] = bool(
                        clone["directional_called"] and clone["prediction"] == clone.get("truth")
                    )
                    clone["correct"] = bool(
                        clone["prediction"] == clone.get("truth") and clone.get("truth") != "Unknown"
                    )
                    clone["bull_called"] = clone["prediction"] == "Bull"
                    clone["bear_called"] = clone["prediction"] == "Bear"
                    clone["neutral_called"] = clone["prediction"] == "Neutral"

                    clone["bull_called_bear_truth"] = bool(clone["bull_called"] and clone.get("truth") == "Bear")
                    clone["bear_called_bull_truth"] = bool(clone["bear_called"] and clone.get("truth") == "Bull")
                    clone["neutral_called_bull_truth"] = bool(clone["neutral_called"] and clone.get("truth") == "Bull")
                    clone["neutral_called_bear_truth"] = bool(clone["neutral_called"] and clone.get("truth") == "Bear")
                    clone["bull_called_neutral_truth"] = bool(clone["bull_called"] and clone.get("truth") == "Neutral")
                    clone["bear_called_neutral_truth"] = bool(clone["bear_called"] and clone.get("truth") == "Neutral")

                simulated.append(clone)

            sim_summary = build_main_summary(simulated)
            new_directional_accuracy = sim_summary["directional_accuracy"]
            new_directional_coverage = sim_summary["directional_coverage"]
            new_bull_precision = sim_summary["bull_precision"]
            new_bear_precision = sim_summary["bear_precision"]

            results.append(
                {
                    "variant": variant,
                    "flip_score_threshold": f_thr,
                    "alignment_threshold": a_thr,
                    "compression_velocity_min": c_min,
                    "ratio_threshold": ratio_thr,
                    "changed_rows_count": len(candidates),
                    "new_directional_called": sim_summary["directional_called"],
                    **baseline,
                    "new_directional_accuracy": new_directional_accuracy,
                    "new_directional_coverage": new_directional_coverage,
                    "new_bull_precision": new_bull_precision,
                    "new_bear_precision": new_bear_precision,
                    "delta_directional_accuracy": new_directional_accuracy - baseline["baseline_directional_accuracy"],
                    "delta_directional_coverage": new_directional_coverage - baseline["baseline_directional_coverage"],
                    "delta_bull_precision": new_bull_precision - baseline["baseline_bull_precision"],
                    "delta_bear_precision": new_bear_precision - baseline["baseline_bear_precision"],
                }
            )

    top_neutral = rank_v2_1_candidates(results, "bull_to_neutral", top_n=5)
    top_bear = rank_v2_1_candidates(results, "bull_to_bear", top_n=3)
    return results, baseline, top_neutral, top_bear


def rank_v2_1_candidates(
    rows: Sequence[Dict[str, Any]],
    variant: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    """
    Rank v2_1 candidates by:
    1) higher directional accuracy gain
    2) higher bear precision gain
    3) less coverage loss
    4) fewer changed rows (small preference)
    """
    subset = [r for r in rows if r.get("variant") == variant]
    subset.sort(
        key=lambda r: (
            _as_float(r.get("delta_directional_accuracy")) or 0.0,
            _as_float(r.get("delta_bear_precision")) or 0.0,
            _as_float(r.get("delta_directional_coverage")) or -1.0,
            -(_as_int(r.get("changed_rows_count")) or 0),
        ),
        reverse=True,
    )
    return subset[:top_n]


def _scan_thr_label(value: Optional[float], none_label: str = "no filter") -> str:
    return none_label if value is None else f"{value:.2f}"


def print_v2_1_scan_summary(
    baseline: Dict[str, Any],
    top_neutral: Sequence[Dict[str, Any]],
    top_bear: Sequence[Dict[str, Any]],
) -> None:
    """
    Print focused v2_1 Bull-run instability diagnostics and ranked candidates.
    """
    print("\n=== V2_1 Bull-Run Instability Scan ===")
    print(
        "baseline directional called/accuracy/coverage: "
        f"{int(baseline['baseline_directional_called'])} / "
        f"{_pct(baseline['baseline_directional_accuracy'])} / "
        f"{_pct(baseline['baseline_directional_coverage'])}"
    )
    print(
        "baseline bull precision / bear precision: "
        f"{_pct(baseline['baseline_bull_precision'])} / "
        f"{_pct(baseline['baseline_bear_precision'])}"
    )

    def _print_candidates(title: str, candidates: Sequence[Dict[str, Any]]) -> None:
        print(f"\n{title}")
        if not candidates:
            print("- none")
            return

        for row in candidates:
            print(
                "- flip>= {flip} | align<= {align} | comp>= {comp} | ratio<= {ratio} | changed= {changed} | "
                "acc= {acc} ({d_acc:+.4f}) | cov= {cov} ({d_cov:+.4f}) | "
                "bull= {bull} ({d_bull:+.4f}) | bear= {bear} ({d_bear:+.4f})".format(
                    flip=_scan_thr_label(_as_float(row.get("flip_score_threshold"))),
                    align=_scan_thr_label(_as_float(row.get("alignment_threshold"))),
                    comp=_scan_thr_label(_as_float(row.get("compression_velocity_min"))),
                    ratio=_scan_thr_label(_as_float(row.get("ratio_threshold"))),
                    changed=int(_as_int(row.get("changed_rows_count")) or 0),
                    acc=_pct(_as_float(row.get("new_directional_accuracy")) or 0.0),
                    d_acc=_as_float(row.get("delta_directional_accuracy")) or 0.0,
                    cov=_pct(_as_float(row.get("new_directional_coverage")) or 0.0),
                    d_cov=_as_float(row.get("delta_directional_coverage")) or 0.0,
                    bull=_pct(_as_float(row.get("new_bull_precision")) or 0.0),
                    d_bull=_as_float(row.get("delta_bull_precision")) or 0.0,
                    bear=_pct(_as_float(row.get("new_bear_precision")) or 0.0),
                    d_bear=_as_float(row.get("delta_bear_precision")) or 0.0,
                )
            )

    _print_candidates("Top 5 Bull -> Neutral candidates", top_neutral)
    _print_candidates("Top 3 Bull -> Bear candidates", top_bear)

def _metric_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    directional = [r for r in rows if r.get("prediction") in {"Bull", "Bear"}]
    correct = [r for r in directional if r.get("prediction") == r.get("truth")]
    bull_rows = [r for r in directional if r.get("prediction") == "Bull"]
    bear_rows = [r for r in directional if r.get("prediction") == "Bear"]
    return {
        "rows": len(rows),
        "reversal_rate": _safe_ratio(sum(1 for r in rows if _is_transition_reversal(r)), len(rows)),
        "continuation_rate": _safe_ratio(sum(1 for r in rows if not _is_transition_reversal(r)), len(rows)),
        "directional_call_rate": _safe_ratio(len(directional), len(rows)),
        "directional_accuracy": _safe_ratio(len(correct), len(directional)),
        "bull_precision": _safe_ratio(sum(1 for r in bull_rows if r.get("truth") == "Bull"), len(bull_rows)),
        "bear_precision": _safe_ratio(sum(1 for r in bear_rows if r.get("truth") == "Bear"), len(bear_rows)),
    }


def build_transition_signal_report(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    signals = [
        "hinge_torsion_spike", "phase_disagreement", "fan_polarity_inversion", "fan_velocity_oscillation",
        "fan_collapse_v2", "fan_collapse_v2_strict", "compression_trap", "compression_trap_strict",
        "fan_energy_release_signal", "price_diff_vol_squeeze",
    ]
    out=[]
    for sig in signals:
        subset=[r for r in rows if (_as_int(r.get(sig)) or 0)==1]
        m=_metric_row(subset)
        out.append({"signal":sig,"row_count":m["rows"],"reversal_rate":m["reversal_rate"],"continuation_rate":m["continuation_rate"],"directional_call_rate":m["directional_call_rate"],"directional_accuracy_within_signal":m["directional_accuracy"]})
    return out


def build_transition_score_report(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out=[]
    for thr in [1,2,3,4]:
        subset=[]
        for r in rows:
            score = (_as_int(r.get("phase_disagreement")) or 0) + (_as_int(r.get("hinge_torsion_spike")) or 0) + (_as_int(r.get("fan_polarity_inversion")) or 0) + (_as_int(r.get("fan_collapse_v2")) or 0)
            if score >= thr:
                subset.append(r)
        m=_metric_row(subset)
        out.append({"threshold":f"score>={thr}","rows":m["rows"],"reversal_rate":m["reversal_rate"],"continuation_rate":m["continuation_rate"],"directional_accuracy":m["directional_accuracy"],"bull_precision":m["bull_precision"],"bear_precision":m["bear_precision"],"coverage_if_used_as_transition_gate":_safe_ratio(len(subset),len(rows))})
    return out


def build_trend_strength_report(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bins=[("0_to_6",0,6),("6_to_15",6,15),("15_to_25",15,25),("25_plus",25,10**9)]
    out=[]
    for name,lo,hi in bins:
        subset=[r for r in rows if (_as_float(r.get("outer_fan_width")) is not None and _as_float(r.get("outer_fan_width"))>=lo and _as_float(r.get("outer_fan_width"))<hi)]
        m=_metric_row(subset)
        out.append({"outer_fan_width_bin":name,"rows":len(subset),"bull_count":sum(1 for r in subset if r.get("prediction")=="Bull"),"bear_count":sum(1 for r in subset if r.get("prediction")=="Bear"),"neutral_count":sum(1 for r in subset if r.get("prediction")=="Neutral"),"directional_accuracy":m["directional_accuracy"],"reversal_rate":m["reversal_rate"],"continuation_rate":m["continuation_rate"]})
    return out


def build_oscillation_report(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    patterns=["+-+","-+-","++-","+--","-++","--+","+++","---"]
    observed={str(r.get("fan_velocity_pattern_3") or "000") for r in rows}
    all_patterns=sorted(observed.union(patterns))
    out=[]
    for pat in all_patterns:
        subset=[r for r in rows if str(r.get("fan_velocity_pattern_3") or "000")==pat]
        m=_metric_row(subset)
        out.append({"pattern":pat,"rows":len(subset),"reversal_rate":m["reversal_rate"],"continuation_rate":m["continuation_rate"],"directional_accuracy":m["directional_accuracy"],"mean_fan_velocity_oscillation_strength":_safe_mean(subset,"fan_velocity_oscillation_strength")})
    return out


def build_price_structure_report(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out=[]
    for pat in sorted({str(r.get("price_diff_sign_pattern") or "000") for r in rows}):
        subset=[r for r in rows if str(r.get("price_diff_sign_pattern") or "000")==pat]
        m=_metric_row(subset)
        out.append({"group":"price_diff_sign_pattern","value":pat,**m})
    for key in ["price_diff_vol_squeeze","direction_flip"]:
        for val in [0,1]:
            subset=[r for r in rows if (_as_int(r.get(key)) or 0)==val]
            m=_metric_row(subset)
            out.append({"group":key,"value":str(val),**m})
    for name,key,bins in [("epoch_volatility","epoch_volatility",[(0,0.5),(0.5,1.0),(1.0,2.0),(2.0,10**9)]),("abs_momentum_3epoch","momentum_3epoch",[(0,0.5),(0.5,1.0),(1.0,2.0),(2.0,10**9)])]:
        for lo,hi in bins:
            subset=[]
            for r in rows:
                v=_as_float(r.get(key))
                if v is None:
                    continue
                if name.startswith("abs"):
                    v=abs(v)
                if lo<=v<hi:
                    subset.append(r)
            m=_metric_row(subset)
            out.append({"group":name,"value":f"{lo}-{hi if hi<10**8 else 'plus'}",**m})
    return out


def simulate_proto_v2_3_offline(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    simulated=[]
    counts={"transition_score_triggers":0,"fan_collapse_v2_triggers":0,"compression_trap_blocks":0,"inertia_wall_neutralizations":0,"oscillation_based_reversals":0}
    for r in rows:
        clone=dict(r)
        pred=_resolve_directional_prediction(r,"prediction")
        base_pred=pred
        fwv=_as_float(r.get("fan_width_velocity")) or 0.0
        transition_score=(_as_int(r.get("transition_score_base")) or 0)+(_as_int(r.get("fan_collapse_v2")) or 0)
        inertia=_as_float(r.get("inertia_wall_score")) or 0.0
        if (_as_int(r.get("compression_trap")) or 0)==1:
            counts["compression_trap_blocks"]+=1
            proto=base_pred if base_pred in {"Bull","Bear"} else "Neutral"
        elif transition_score >= 2 and abs(fwv) >= 20:
            counts["transition_score_triggers"]+=1
            proto="Bull" if fwv>0 else "Bear"
            if (_as_int(r.get("fan_velocity_oscillation")) or 0)==1:
                counts["oscillation_based_reversals"]+=1
        elif (_as_int(r.get("fan_collapse_v2")) or 0)==1:
            counts["fan_collapse_v2_triggers"]+=1
            proto="Bear" if fwv<0 else "Bull"
        else:
            proto=base_pred
        if inertia >= INERTIA_WALL_HIGH_THRESHOLD and proto in {"Bull","Bear"} and proto != base_pred:
            counts["inertia_wall_neutralizations"] += 1
            proto = "Neutral"
        clone["proto_v2_3_prediction"]=proto
        clone["prediction"]=proto
        clone["directional_called"]=proto in {"Bull","Bear"}
        clone["directional_correct"]=bool(clone["directional_called"] and proto==clone.get("truth"))
        simulated.append(clone)
    s=build_main_summary(simulated)
    report={"total_rows":len(simulated),"directional_called":s.get("directional_called",0),"neutral_called":s.get("neutral_called",0),"directional_accuracy":s.get("directional_accuracy",0.0),"coverage":s.get("directional_coverage",0.0),"bull_precision":s.get("bull_precision",0.0),"bear_precision":s.get("bear_precision",0.0),"bull_pred_bull_truth":s.get("bull_pred_bull_truth",0),"bull_pred_bear_truth":s.get("bull_pred_bear_truth",0),"bear_pred_bull_truth":s.get("bear_pred_bull_truth",0),"bear_pred_bear_truth":s.get("bear_pred_bear_truth",0),**counts}
    return {"rows":simulated,"report":report}

def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    regime_breakdown: Sequence[Dict[str, Any]],
    error_breakdown: Sequence[Dict[str, Any]],
    run_flip_breakdown: Sequence[Dict[str, Any]],
    threshold_scan: Sequence[Dict[str, Any]],
    v2_1_bull_instability_scan: Sequence[Dict[str, Any]],
    fan_feature_diagnostics: Sequence[Dict[str, Any]],
    v2_2_feature_candidates: Sequence[Dict[str, Any]],
    feature_population_audit: Sequence[Dict[str, Any]],
    transition_signal_report: Sequence[Dict[str, Any]],
    transition_score_report: Sequence[Dict[str, Any]],
    trend_strength_report: Sequence[Dict[str, Any]],
    oscillation_report: Sequence[Dict[str, Any]],
    price_structure_report: Sequence[Dict[str, Any]],
    proto_v2_3_offline_report: Sequence[Dict[str, Any]],
    model_suffix: str,
) -> Dict[str, Path]:
    """Write all required datasets and summary files."""
    paths = {
        "dataset": output_dir / f"gaussian_fan_dataset_{model_suffix}.csv",
        "wrong": output_dir / f"gaussian_fan_wrong_predictions_{model_suffix}.csv",
        "neutral_misses": output_dir / f"gaussian_fan_neutral_misses_{model_suffix}.csv",
        "summary": output_dir / f"gaussian_fan_summary_{model_suffix}.json",
        "regime_breakdown": output_dir / f"gaussian_fan_regime_breakdown_{model_suffix}.csv",
        "error_breakdown": output_dir / f"gaussian_fan_error_breakdown_{model_suffix}.csv",
        "run_flip_breakdown": output_dir / f"gaussian_fan_run_flip_breakdown_{model_suffix}.csv",
        "threshold_scan": output_dir / f"gaussian_fan_threshold_scan_{model_suffix}.csv",
        "v2_1_bull_instability_scan": output_dir / f"gaussian_fan_v2_1_bull_instability_scan_{model_suffix}.csv",
        "fan_feature_diagnostics": output_dir / f"gaussian_fan_feature_diagnostics_{model_suffix}.csv",
        "v2_2_feature_candidates": output_dir / f"gaussian_fan_v2_2_feature_candidates_{model_suffix}.csv",
        "feature_population_audit": output_dir / f"gaussian_fan_feature_population_audit_{model_suffix}.csv",
        "wave_inertia_profile": output_dir / f"gaussian_fan_wave_inertia_profile_{model_suffix}.csv",
        "feature_source_map": output_dir / "gaussian_fan_feature_source_map.txt",
        "transition_signal_report": output_dir / f"gaussian_fan_transition_signal_report_{model_suffix}.csv",
        "transition_score_report": output_dir / f"gaussian_fan_transition_score_report_{model_suffix}.csv",
        "trend_strength_report": output_dir / f"gaussian_fan_trend_strength_report_{model_suffix}.csv",
        "oscillation_report": output_dir / f"gaussian_fan_oscillation_report_{model_suffix}.csv",
        "price_structure_report": output_dir / f"gaussian_fan_price_structure_report_{model_suffix}.csv",
        "proto_v2_3_offline_report": output_dir / f"gaussian_fan_proto_v2_3_offline_report_{model_suffix}.csv",
    }
    if model_suffix == "v2_1":
        paths["v2_2_feature_candidates"] = output_dir / "gaussian_fan_v2_2_feature_candidates_v2_1.csv"

    wrong_rows = [r for r in rows if r.get("directional_called") and not r.get("directional_correct")]
    neutral_misses = [r for r in rows if r.get("prediction") == "Neutral" and r.get("truth") in {"Bull", "Bear"}]

    _write_csv(paths["dataset"], rows, CSV_COLUMNS)
    _write_csv(paths["wrong"], wrong_rows, CSV_COLUMNS)
    _write_csv(paths["neutral_misses"], neutral_misses, CSV_COLUMNS)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    group_fields = list(regime_breakdown[0].keys()) if regime_breakdown else ["group_name", "group_value", "row_count"]
    _write_csv(paths["regime_breakdown"], regime_breakdown, group_fields)
    _write_csv(paths["error_breakdown"], error_breakdown, list(error_breakdown[0].keys()) if error_breakdown else group_fields)
    _write_csv(paths["run_flip_breakdown"], run_flip_breakdown, list(run_flip_breakdown[0].keys()) if run_flip_breakdown else group_fields)
    _write_csv(paths["threshold_scan"], threshold_scan, list(threshold_scan[0].keys()) if threshold_scan else ["variant"])
    _write_csv(
        paths["v2_1_bull_instability_scan"],
        v2_1_bull_instability_scan,
        list(v2_1_bull_instability_scan[0].keys()) if v2_1_bull_instability_scan else ["variant"],
    )
    _write_csv(
        paths["fan_feature_diagnostics"],
        fan_feature_diagnostics,
        list(fan_feature_diagnostics[0].keys()) if fan_feature_diagnostics else ["group_name", "group_value", "row_count"],
    )
    _write_csv(
        paths["v2_2_feature_candidates"],
        v2_2_feature_candidates,
        list(v2_2_feature_candidates[0].keys()) if v2_2_feature_candidates else ["group_name", "group_value", "row_count"],
    )
    write_feature_population_audit(paths["feature_population_audit"], feature_population_audit)
    wave_inertia_profile = build_wave_inertia_profile(rows)
    _write_csv(
        paths["wave_inertia_profile"],
        wave_inertia_profile,
        list(wave_inertia_profile[0].keys()) if wave_inertia_profile else ["directional_error_type", "row_count"],
    )
    _write_csv(paths["transition_signal_report"], transition_signal_report, list(transition_signal_report[0].keys()) if transition_signal_report else ["signal", "row_count"])
    _write_csv(paths["transition_score_report"], transition_score_report, list(transition_score_report[0].keys()) if transition_score_report else ["threshold", "rows"])
    _write_csv(paths["trend_strength_report"], trend_strength_report, list(trend_strength_report[0].keys()) if trend_strength_report else ["outer_fan_width_bin", "rows"])
    _write_csv(paths["oscillation_report"], oscillation_report, list(oscillation_report[0].keys()) if oscillation_report else ["pattern", "rows"])
    _write_csv(paths["price_structure_report"], price_structure_report, list(price_structure_report[0].keys()) if price_structure_report else ["group", "value", "rows"])
    _write_csv(paths["proto_v2_3_offline_report"], proto_v2_3_offline_report, list(proto_v2_3_offline_report[0].keys()) if proto_v2_3_offline_report else ["total_rows"])

    return paths


def _pct(value: float) -> str:
    return f"{(value or 0.0) * 100:.2f}%"

def print_terminal_summary(
    model_id: str,
    summary: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    error_breakdown: Sequence[Dict[str, Any]],
    paths: Dict[str, Path],
    v2_1_scan_baseline: Dict[str, Any],
    top_bull_to_neutral_candidates: Sequence[Dict[str, Any]],
    top_bull_to_bear_candidates: Sequence[Dict[str, Any]],
    feature_population_counts: Dict[str, Any],
    population_warnings: Sequence[str],
    show_scan: bool,
) -> None:
    """Print readable diagnostics sections in terminal."""
    print(f"\n=== {model_id} Results ===")
    print("=== Gaussian Fan Export: Headline Stats ===")
    print(f"Total Rows: {summary['total_rows']}")
    print(f"Directional Called: {summary['directional_called']}")
    print(f"Neutral Called: {summary['neutral_called']}")
    print(f"Directional Accuracy: {_pct(summary['directional_accuracy'])}")
    print(f"Directional Coverage: {_pct(summary['directional_coverage'])}")
    print(f"Neutral Rate: {_pct(summary['neutral_rate'])}")
    print(f"Bull Precision: {_pct(summary['bull_precision'])}")
    print(f"Bear Precision: {_pct(summary['bear_precision'])}")

    print("\n=== Confusion-style Counts ===")
    print(f"Bull predicted / Bull truth: {summary['bull_pred_bull_truth']}")
    print(f"Bull predicted / Bear truth: {summary['bull_pred_bear_truth']}")
    print(f"Bear predicted / Bull truth: {summary['bear_pred_bull_truth']}")
    print(f"Bear predicted / Bear truth: {summary['bear_pred_bear_truth']}")
    print(f"Neutral predicted / Bull truth: {summary['neutral_pred_bull_truth']}")
    print(f"Neutral predicted / Bear truth: {summary['neutral_pred_bear_truth']}")

    print("\n=== Regime Breakdown Summary ===")
    print(f"RUN rows / accuracy: {summary['run_rows']} / {_pct(summary['run_directional_accuracy'])}")
    print(f"REVERSAL rows / accuracy: {summary['reversal_rows']} / {_pct(summary['reversal_directional_accuracy'])}")
    print(f"NOISE rows / accuracy: {summary['noise_rows']} / {_pct(summary['noise_directional_accuracy'])}")

    wrong_bull = next((r for r in error_breakdown if r.get("group_value") == "v2_1_wrong_bull_to_bear"), None)
    correct_bull = next((r for r in error_breakdown if r.get("group_value") == "v2_1_correct_bull"), None)
    print("\n=== V2_1 Bear Improvement Diagnostic ===")
    print(f"wrong Bull->Bear rows: {int((wrong_bull or {}).get('row_count', 0))}")
    print(f"correct Bull rows: {int((correct_bull or {}).get('row_count', 0))}")
    if wrong_bull and correct_bull:
        print(
            "mean compression_velocity (wrong vs correct): "
            f"{wrong_bull.get('mean_compression_velocity', 0.0):.5f} vs {correct_bull.get('mean_compression_velocity', 0.0):.5f}"
        )
        print(
            "mean alignment (wrong vs correct): "
            f"{wrong_bull.get('mean_alignment', 0.0):.5f} vs {correct_bull.get('mean_alignment', 0.0):.5f}"
        )
        print(
            "mean flip_score (wrong vs correct): "
            f"{wrong_bull.get('mean_flip_score', 0.0):.5f} vs {correct_bull.get('mean_flip_score', 0.0):.5f}"
        )
        print(
            "weak_bull_exhaustion_flag rate (wrong vs correct): "
            f"{_pct(wrong_bull.get('weak_bull_exhaustion_flag_rate', 0.0))} vs "
            f"{_pct(correct_bull.get('weak_bull_exhaustion_flag_rate', 0.0))}"
        )
        print(
            "mean ratio_8_23_to_23_53 (wrong vs correct): "
            f"{wrong_bull.get('mean_ratio_8_23_to_23_53', 0.0):.5f} vs {correct_bull.get('mean_ratio_8_23_to_23_53', 0.0):.5f}"
        )
        print(
            "mean slope_sum (wrong vs correct): "
            f"{wrong_bull.get('mean_slope_sum', 0.0):.5f} vs {correct_bull.get('mean_slope_sum', 0.0):.5f}"
        )
        print(
            "mean slope_magnitude (wrong vs correct): "
            f"{wrong_bull.get('mean_slope_magnitude', 0.0):.5f} vs {correct_bull.get('mean_slope_magnitude', 0.0):.5f}"
        )
        print(
            "slope_disagreement_rate (wrong vs correct): "
            f"{_pct(wrong_bull.get('slope_disagreement_rate', 0.0))} vs {_pct(correct_bull.get('slope_disagreement_rate', 0.0))}"
        )

    preview = summary.get("v2_2_feature_preview", {}) if isinstance(summary.get("v2_2_feature_preview"), dict) else {}
    geom = preview.get("geometry", {}) if isinstance(preview.get("geometry"), dict) else {}
    energy = preview.get("energy", {}) if isinstance(preview.get("energy"), dict) else {}
    hinge = preview.get("hinge", {}) if isinstance(preview.get("hinge"), dict) else {}
    snap = preview.get("snap", {}) if isinstance(preview.get("snap"), dict) else {}
    phase = preview.get("phase", {}) if isinstance(preview.get("phase"), dict) else {}
    wave_inertia = preview.get("wave_inertia", {}) if isinstance(preview.get("wave_inertia"), dict) else {}

    print("\n=== v2_2 Feature Preview ===")
    print(f"mean fan_width: {geom.get('mean_fan_width', 0.0):.5f}")
    print(f"mean fan_energy_instability: {energy.get('mean_fan_energy_instability', 0.0):.5f}")
    print(f"hinge_conflict_rate: {_pct(hinge.get('hinge_conflict_rate', 0.0) or 0.0)}")
    print(f"mean snap_score: {snap.get('mean_snap_score', 0.0):.5f}")
    print(f"mean fan_phase_score: {phase.get('mean_fan_phase_score', 0.0):.5f}")

    print("\n=== v2_3 Wave / Inertia Preview ===")
    print(f"gaussian_wave_rate: {_pct(wave_inertia.get('gaussian_wave_rate', 0.0) or 0.0)}")
    print(f"mean wave_strength: {wave_inertia.get('mean_wave_strength', 0.0):.5f}")
    print(f"mean slow_retention: {wave_inertia.get('mean_slow_retention', 0.0):.5f}")
    print(f"phase_disagreement_rate: {_pct(wave_inertia.get('phase_disagreement_rate', 0.0) or 0.0)}")
    print(f"mean inertia_wall_score: {wave_inertia.get('mean_inertia_wall_score', 0.0):.5f}")

    print("\n=== v2_3 Transition / Collapse Preview ===")
    print(f"hinge_torsion_spike rate: {_pct(_safe_mean(rows, 'hinge_torsion_spike'))}")
    print(f"fan_polarity_inversion rate: {_pct(_safe_mean(rows, 'fan_polarity_inversion'))}")
    print(f"fan_velocity_oscillation rate: {_pct(_safe_mean(rows, 'fan_velocity_oscillation'))}")
    print(f"fan_collapse_v2 rate: {_pct(_safe_mean(rows, 'fan_collapse_v2'))}")
    print(f"compression_trap rate: {_pct(_safe_mean(rows, 'compression_trap'))}")
    print(f"price_diff_vol_squeeze rate: {_pct(_safe_mean(rows, 'price_diff_vol_squeeze'))}")
    print(f"fan_energy_release_signal rate: {_pct(_safe_mean(rows, 'fan_energy_release_signal'))}")
    print(f"transition_score>=2 rate: {_pct(_safe_ratio(sum(1 for r in rows if (_as_int(r.get('transition_score_base')) or 0) >= 2), len(rows)))}")

    proto = summary.get("proto_v2_3_offline", {}) if isinstance(summary.get("proto_v2_3_offline"), dict) else {}
    print("\n=== proto_v2_3 Offline Preview ===")
    print(f"directional accuracy: {_pct(proto.get('directional_accuracy', 0.0) or 0.0)}")
    print(f"coverage: {_pct(proto.get('coverage', 0.0) or 0.0)}")
    print(f"bull precision: {_pct(proto.get('bull_precision', 0.0) or 0.0)}")
    print(f"bear precision: {_pct(proto.get('bear_precision', 0.0) or 0.0)}")
    print(f"compression_trap blocks: {int(proto.get('compression_trap_blocks', 0) or 0)}")
    print(f"fan_collapse promotions: {int(proto.get('fan_collapse_v2_triggers', 0) or 0)}")
    print(f"transition_score triggers: {int(proto.get('transition_score_triggers', 0) or 0)}")
    print(f"inertia neutralizations: {int(proto.get('inertia_wall_neutralizations', 0) or 0)}")

    if model_id == MODEL_V2_1:
        wrong_bull = next((r for r in error_breakdown if r.get("group_value") == "bull_called_bear_truth"), None)
        correct_bull = next((r for r in error_breakdown if r.get("group_value") == "correct_bull"), None)
        wrong_bear = next((r for r in error_breakdown if r.get("group_value") == "bear_called_bull_truth"), None)
        correct_bear = next((r for r in error_breakdown if r.get("group_value") == "correct_bear"), None)
        print("\n=== v2_2 Focused Comparison (v2_1) ===")
        if wrong_bull and correct_bull:
            print(
                "bull_called_bear_truth vs correct_bull | "
                f"fan_order={wrong_bull.get('mean_fan_order_score', 0.0):.4f} vs {correct_bull.get('mean_fan_order_score', 0.0):.4f} | "
                f"energy_instability={wrong_bull.get('mean_fan_energy_instability', 0.0):.4f} vs {correct_bull.get('mean_fan_energy_instability', 0.0):.4f}"
            )
        if wrong_bear and correct_bear:
            print(
                "bear_called_bull_truth vs correct_bear | "
                f"hinge_torsion={wrong_bear.get('mean_hinge_torsion', 0.0):.4f} vs {correct_bear.get('mean_hinge_torsion', 0.0):.4f} | "
                f"phase_score={wrong_bear.get('mean_fan_phase_score', 0.0):.4f} vs {correct_bear.get('mean_fan_phase_score', 0.0):.4f}"
            )

    print("\n=== Advanced Feature Population Audit ===")
    print(
        "sigma population: "
        + " | ".join(
            f"{k}={int(feature_population_counts.get(f'rows_with_{k}', 0))}/{int(feature_population_counts.get('total_rows', 0))}"
            for k in ["g8", "g23", "g38", "g53", "g68", "g83"]
        )
    )
    print(
        "slope population: "
        + " | ".join(
            f"{k}={int(feature_population_counts.get(f'rows_with_{k}', 0))}/{int(feature_population_counts.get('total_rows', 0))}"
            for k in ["slope_g8", "slope_g23", "slope_g38", "slope_g53", "slope_g68", "slope_g83"]
        )
    )
    print(
        "advanced population: "
        + " | ".join(
            f"{k}={int(feature_population_counts.get(f'rows_with_{k}', 0))}"
            for k in ["fan_width", "fan_order_score", "g83_curvature", "fan_energy_total", "fan_energy_ratio", "hinge_gap", "hinge_torsion", "snap_score", "phase_alignment", "fan_phase_score"]
        )
    )
    print(
        "non-zero counts: "
        f"fan_order_score={int(feature_population_counts.get('rows_with_nonzero_fan_order_score', 0))} | "
        f"hinge_torsion={int(feature_population_counts.get('rows_with_nonzero_hinge_torsion', 0))} | "
        f"snap_score={int(feature_population_counts.get('rows_with_nonzero_snap_score', 0))} | "
        f"phase_alignment={int(feature_population_counts.get('rows_with_nonzero_phase_alignment', 0))}"
    )
    for warn in population_warnings:
        print(warn)

    if show_scan:
        print_v2_1_scan_summary(
            v2_1_scan_baseline,
            top_bull_to_neutral_candidates,
            top_bull_to_bear_candidates,
        )

    print("\n=== Saved Files ===")
    for key in [
        "dataset",
        "wrong",
        "neutral_misses",
        "summary",
        "regime_breakdown",
        "error_breakdown",
        "run_flip_breakdown",
        "threshold_scan",
        "v2_1_bull_instability_scan",
        "fan_feature_diagnostics",
        "v2_2_feature_candidates",
        "feature_population_audit",
        "wave_inertia_profile",
        "transition_signal_report",
        "transition_score_report",
        "trend_strength_report",
        "oscillation_report",
        "price_structure_report",
        "proto_v2_3_offline_report",
    ]:
        print(f"- {paths[key].resolve()}")


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _summary_slice(summary: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "directional_accuracy",
        "directional_coverage",
        "neutral_rate",
        "bull_precision",
        "bear_precision",
        "directional_called",
        "total_rows",
    ]
    return {k: summary.get(k, 0.0) for k in keys}


def build_model_comparison(summary_v2_0: Dict[str, Any], summary_v2_1: Dict[str, Any], summary_v2_2: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    deltas_v21 = {}
    for key in ["directional_accuracy", "directional_coverage", "neutral_rate", "bull_precision", "bear_precision"]:
        deltas_v21[f"delta_{key}"] = (summary_v2_1.get(key, 0.0) or 0.0) - (summary_v2_0.get(key, 0.0) or 0.0)
    out = {
        MODEL_V2_0: _summary_slice(summary_v2_0),
        MODEL_V2_1: _summary_slice(summary_v2_1),
        "delta_v2_1_minus_v2_0": deltas_v21,
    }
    if isinstance(summary_v2_2, dict):
        deltas_v22 = {}
        for key in ["directional_accuracy", "directional_coverage", "neutral_rate", "bull_precision", "bear_precision"]:
            deltas_v22[f"delta_{key}"] = (summary_v2_2.get(key, 0.0) or 0.0) - (summary_v2_1.get(key, 0.0) or 0.0)
        out[MODEL_V2_2] = _summary_slice(summary_v2_2)
        out["delta_v2_2_minus_v2_1"] = deltas_v22
    return out

def build_v2_1_override_impact(
    rows_v2_0: Sequence[Dict[str, Any]],
    rows_v2_1: Sequence[Dict[str, Any]],
    summary_v2_0: Dict[str, Any],
    summary_v2_1: Dict[str, Any],
) -> Dict[str, Any]:
    v2_0_map = {(r.get("epoch"), r.get("next_epoch"), r.get("timestamp")): r for r in rows_v2_0}
    v2_1_map = {(r.get("epoch"), r.get("next_epoch"), r.get("timestamp")): r for r in rows_v2_1}

    override_rows = []
    became_correct = 0
    became_incorrect = 0
    for key, r1 in v2_1_map.items():
        r0 = v2_0_map.get(key)
        if not r0:
            continue
        if r0.get("prediction") == "Bull" and r1.get("prediction") == "Bear":
            override_rows.append((r0, r1))
            truth = r1.get("truth")
            if truth == "Bear":
                became_correct += 1
            elif truth == "Bull":
                became_incorrect += 1

    comparison = build_model_comparison(summary_v2_0, summary_v2_1)
    return {
        "bull_to_bear_overrides_applied": len(override_rows),
        "bull_to_bear_overrides_became_correct": became_correct,
        "bull_to_bear_overrides_became_incorrect": became_incorrect,
        "change_vs_v2_0": comparison.get("delta_v2_1_minus_v2_0", {}),
    }

def build_feature_diagnostics(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    numeric = [
        "fan_width_velocity","hinge_torsion","outer_fan_width","epoch_volatility","momentum_3epoch",
        "fan_velocity_oscillation_strength","fan_energy_total"
    ]
    binary = [
        "hinge_torsion_spike","fan_polarity_inversion","fan_velocity_oscillation","fan_collapse_v2",
        "compression_trap","price_diff_vol_squeeze","fan_energy_release_signal"
    ]
    patterns = ["fan_velocity_pattern_3","price_diff_sign_pattern","fan_energy_cycle_stage"]
    for key in numeric:
        vals=[_as_float(r.get(key)) for r in rows]
        valid=[v for v in vals if v is not None]
        out.append({"group_name":"numeric_summary","group_value":key,"row_count":len(valid),"mean":float(mean(valid)) if valid else 0.0,"std":0.0 if len(valid)<2 else float((sum((v-(sum(valid)/len(valid)))**2 for v in valid)/len(valid))**0.5),"min":min(valid) if valid else 0.0,"max":max(valid) if valid else 0.0})
    for key in binary:
        out.append({"group_name":"binary_rate","group_value":key,"row_count":len(rows),"rate":_safe_ratio(sum(1 for r in rows if (_as_int(r.get(key)) or 0)==1),len(rows))})
    for key in patterns:
        vals=[str(r.get(key) or "") for r in rows]
        for value in sorted(set(vals)):
            out.append({"group_name":"pattern_frequency","group_value":f"{key}:{value}","row_count":sum(1 for v in vals if v==value),"rate":_safe_ratio(sum(1 for v in vals if v==value),len(vals))})
    return out

def run_model_export(
    model_id: str,
    history_rows: Sequence[Dict[str, Any]],
    round_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    flat_rows = flatten_model_rows(history_rows, model_id)
    joined = join_truth(flat_rows, round_rows)
    rows = add_derived_columns(joined)

    summary = build_main_summary(rows)
    regime_breakdown = build_regime_breakdown(rows)
    error_breakdown = build_error_breakdown(rows)
    run_flip_breakdown = build_run_flip_breakdown(rows)
    fan_feature_diagnostics = []
    fan_feature_diagnostics.extend(_group_rows(rows, "prediction", ["Bull", "Bear", "Neutral", "Unknown"]))
    fan_feature_diagnostics.extend(_group_rows(rows, "truth", ["Bull", "Bear", "Neutral", "Unknown"]))
    fan_feature_diagnostics.extend(build_regime_breakdown(rows))
    fan_feature_diagnostics.extend(build_error_breakdown(rows))
    fan_feature_diagnostics.extend(build_feature_diagnostics(rows))
    feature_population_audit, feature_population_counts, population_warnings = build_feature_population_audit(rows)
    v2_2_feature_candidates = build_v2_2_feature_candidates_v2_1(rows)
    v2_1_scan_rows, v2_1_scan_baseline, top_bull_to_neutral_candidates, top_bull_to_bear_candidates = run_v2_1_bull_instability_scan(rows)
    transition_signal_report = build_transition_signal_report(rows)
    transition_score_report = build_transition_score_report(rows)
    trend_strength_report = build_trend_strength_report(rows)
    oscillation_report = build_oscillation_report(rows)
    price_structure_report = build_price_structure_report(rows)
    proto_sim = simulate_proto_v2_3_offline(rows)
    proto_report_row = proto_sim.get("report", {}) if isinstance(proto_sim, dict) else {}

    summary["v2_1_bull_instability_scan"] = {
        "baseline": v2_1_scan_baseline,
        "top_bull_to_neutral_candidates": top_bull_to_neutral_candidates,
        "top_bull_to_bear_candidates": top_bull_to_bear_candidates,
    }
    summary["v2_2_feature_preview"] = build_v2_2_feature_preview(rows)
    summary["advanced_feature_population_audit"] = feature_population_counts
    summary["advanced_feature_population_warnings"] = population_warnings
    summary["proto_v2_3_offline"] = proto_report_row

    paths = write_outputs(
        output_dir=output_dir,
        rows=rows,
        summary=summary,
        regime_breakdown=regime_breakdown,
        error_breakdown=error_breakdown,
        run_flip_breakdown=run_flip_breakdown,
        threshold_scan=v2_1_scan_rows,
        v2_1_bull_instability_scan=v2_1_scan_rows,
        fan_feature_diagnostics=fan_feature_diagnostics,
        v2_2_feature_candidates=v2_2_feature_candidates,
        feature_population_audit=feature_population_audit,
        transition_signal_report=transition_signal_report,
        transition_score_report=transition_score_report,
        trend_strength_report=trend_strength_report,
        oscillation_report=oscillation_report,
        price_structure_report=price_structure_report,
        proto_v2_3_offline_report=[proto_report_row],
        model_suffix="v2_0" if model_id == MODEL_V2_0 else ("v2_1" if model_id == MODEL_V2_1 else "v2_2"),
    )

    print_terminal_summary(
        model_id,
        summary,
        rows,
        error_breakdown,
        paths,
        v2_1_scan_baseline,
        top_bull_to_neutral_candidates,
        top_bull_to_bear_candidates,
        feature_population_counts,
        population_warnings,
        show_scan=(model_id == MODEL_V2_1),
    )
    return {
        "rows": rows,
        "summary": summary,
        "paths": paths,
        "feature_population_counts": feature_population_counts,
    }

def summarize_v2_2_override_impact(
    rows_v2_1: Sequence[Dict[str, Any]],
    rows_v2_2: Sequence[Dict[str, Any]],
    summary_v2_1: Dict[str, Any],
    summary_v2_2: Dict[str, Any],
) -> Dict[str, Any]:
    counts = {
        "continuation_protection_blocks": 0,
        "collapse_reversal_overrides": 0,
        "false_snap_blocks": 0,
        "inertia_wall_neutrals": 0,
        "reversal_pressure_overrides": 0,
        "neutral_promotions": 0,
    }
    for row in rows_v2_2:
        rule = str(row.get("v2_2_rule") or "")
        if rule == "RUN_CONTINUATION_PROTECTION":
            counts["continuation_protection_blocks"] += 1
        elif rule == "COLLAPSE_REVERSAL":
            counts["collapse_reversal_overrides"] += 1
        elif rule == "FALSE_SNAP_BLOCK":
            counts["false_snap_blocks"] += 1
        elif rule == "INERTIA_WALL_NEUTRAL":
            counts["inertia_wall_neutrals"] += 1
        elif rule == "REVERSAL_PRESSURE_OVERRIDE":
            counts["reversal_pressure_overrides"] += 1
        elif rule == "STRONG_RUN_PROMOTION":
            counts["neutral_promotions"] += 1

    return {
        **counts,
        "v2_2_accuracy_delta_vs_v2_1": (summary_v2_2.get("directional_accuracy", 0.0) or 0.0) - (summary_v2_1.get("directional_accuracy", 0.0) or 0.0),
        "v2_2_bull_precision_delta_vs_v2_1": (summary_v2_2.get("bull_precision", 0.0) or 0.0) - (summary_v2_1.get("bull_precision", 0.0) or 0.0),
        "v2_2_bear_precision_delta_vs_v2_1": (summary_v2_2.get("bear_precision", 0.0) or 0.0) - (summary_v2_1.get("bear_precision", 0.0) or 0.0),
        "v2_2_coverage_delta_vs_v2_1": (summary_v2_2.get("directional_coverage", 0.0) or 0.0) - (summary_v2_1.get("directional_coverage", 0.0) or 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gaussian fan dataset exporter + diagnostics")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH, help="Prediction history JSONL path")
    parser.add_argument("--history-with-v2-2", "--history-with-v2-1", dest="history_with_v2_2", type=Path, default=DEFAULT_REPLAY_HISTORY_PATH,
                        help="Replay-enriched prediction history JSONL output path")
    parser.add_argument("--round", dest="round_path", type=Path, default=DEFAULT_ROUND_PATH,
                        help="Round record JSON path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--v2-1-config", type=Path,
                        default=(SCRIPT_DIR / "models" / "trend_method_v2_1.json").resolve(),
                        help="v2_1 config JSON path")
    parser.add_argument("--v2-2-config", type=Path,
                        default=(SCRIPT_DIR / "models" / "trend_method_v2_2.json").resolve(),
                        help="v2_2 config JSON path")
    args = parser.parse_args()

    history_rows = load_prediction_history(args.history)
    round_rows = load_round_record(args.round_path)

    v2_1_config = load_json_file(args.v2_1_config)
    v2_1_thresholds = v2_1_config.get("thresholds", {}) if isinstance(v2_1_config, dict) else {}
    replay_history_rows, replay_stats_v21 = build_replay_history_with_v2_1(history_rows, v2_1_thresholds)
    v2_2_config = load_json_file(args.v2_2_config)
    replay_history_rows, replay_stats_v22 = build_replay_history_with_v2_2(replay_history_rows, v2_2_config if isinstance(v2_2_config, dict) else {})
    write_prediction_history(args.history_with_v2_2, replay_history_rows)

    print("\n=== Replay Generation ===")
    for k in [
        "rows_seen",
        "rows_with_models",
        "rows_with_v2_0",
        "rows_with_existing_v2_1",
        "rows_with_v2_1_appended",
        "rows_skipped_missing_signals",
        "bull_to_bear_overrides",
    ]:
        print(f"{k}: {replay_stats_v21.get(k, 0)}")
    for k in [
        "rows_with_v2_1",
        "rows_with_existing_v2_2",
        "rows_with_v2_2_appended",
        "rows_skipped_missing_base",
    ]:
        print(f"{k}: {replay_stats_v22.get(k, 0)}")
    print(f"replay_output: {args.history_with_v2_2.resolve()}")

    result_v2_0 = run_model_export(MODEL_V2_0, replay_history_rows, round_rows, args.output_dir)
    result_v2_1 = run_model_export(MODEL_V2_1, replay_history_rows, round_rows, args.output_dir)
    result_v2_2 = run_model_export(MODEL_V2_2, replay_history_rows, round_rows, args.output_dir)
    write_feature_source_map(
        args.output_dir / "gaussian_fan_feature_source_map.txt",
        {
            MODEL_V2_0: result_v2_0.get("feature_population_counts", {}),
            MODEL_V2_1: result_v2_1.get("feature_population_counts", {}),
            MODEL_V2_2: result_v2_2.get("feature_population_counts", {}),
        },
    )
    override_impact = build_v2_1_override_impact(
        result_v2_0["rows"],
        result_v2_1["rows"],
        result_v2_0["summary"],
        result_v2_1["summary"],
    )
    v2_2_impact = summarize_v2_2_override_impact(
        result_v2_1["rows"],
        result_v2_2["rows"],
        result_v2_1["summary"],
        result_v2_2["summary"],
    )

    comparison = build_model_comparison(result_v2_0["summary"], result_v2_1["summary"], result_v2_2["summary"])
    comparison["v2_1_override_impact"] = override_impact
    comparison["v2_2_override_impact"] = v2_2_impact

    comparison_path = args.output_dir / "gaussian_fan_model_comparison.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("\n=== V2_1 Bull-Run Instability Override Impact ===")
    print(f"Bull->Bear overrides applied: {override_impact['bull_to_bear_overrides_applied']}")
    print(f"Bull->Bear overrides became correct: {override_impact['bull_to_bear_overrides_became_correct']}")
    print(f"Bull->Bear overrides became incorrect: {override_impact['bull_to_bear_overrides_became_incorrect']}")
    print(
        f"Directional accuracy delta vs v2_0: {_pct((override_impact['change_vs_v2_0'].get('delta_directional_accuracy') or 0.0))}")
    print(
        f"Bear precision delta vs v2_0: {_pct((override_impact['change_vs_v2_0'].get('delta_bear_precision') or 0.0))}")
    print(
        f"Bull precision delta vs v2_0: {_pct((override_impact['change_vs_v2_0'].get('delta_bull_precision') or 0.0))}")
    print(
        f"Coverage delta vs v2_0: {_pct((override_impact['change_vs_v2_0'].get('delta_directional_coverage') or 0.0))}")
    print("\n=== V2_2 Fan-Physics Override Impact ===")
    for key in [
        "continuation_protection_blocks",
        "collapse_reversal_overrides",
        "false_snap_blocks",
        "inertia_wall_neutrals",
        "reversal_pressure_overrides",
        "neutral_promotions",
    ]:
        print(f"{key}: {v2_2_impact.get(key, 0)}")
    print(f"v2_2 accuracy delta vs v2_1: {_pct(v2_2_impact.get('v2_2_accuracy_delta_vs_v2_1', 0.0))}")
    print(f"v2_2 bull precision delta vs v2_1: {_pct(v2_2_impact.get('v2_2_bull_precision_delta_vs_v2_1', 0.0))}")
    print(f"v2_2 bear precision delta vs v2_1: {_pct(v2_2_impact.get('v2_2_bear_precision_delta_vs_v2_1', 0.0))}")
    print(f"v2_2 coverage delta vs v2_1: {_pct(v2_2_impact.get('v2_2_coverage_delta_vs_v2_1', 0.0))}")
    print(f"comparison_output: {comparison_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())