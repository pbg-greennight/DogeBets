from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class StructureSignal:
    signal: str       # Bull / Bear / Skip
    score: float
    template: str
    reason: str


def _f(row, key: str, default: float = 0.0) -> float:
    val = row.get(key, default)
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def _i(row, key: str, default: int = 0) -> int:
    val = row.get(key, default)
    try:
        if val is None or pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def generate_structure_signal(row, cfg: dict) -> StructureSignal:
    """
    v2:
    Segment-first logic.
    Hard gates first, then small boosters.
    """
    gate = cfg.get("gate", {})

    # --- segment / anchor gates ---
    require_segment_valid = bool(gate.get("require_segment_valid", True))
    min_segment_len = int(gate.get("min_segment_len", 10))
    min_tail_len = int(gate.get("min_tail_len", 5))

    # --- directional gates ---
    min_seg_slope = float(gate.get("min_seg_slope", 0.0))
    min_tail_slope = float(gate.get("min_tail_slope", 0.0))
    min_tail_r2 = float(gate.get("min_tail_r2", 0.20))

    # --- structure quality gates ---
    max_tail_plateau = float(gate.get("max_tail_plateau", 0.55))
    max_tail_fan_inversion = int(gate.get("max_tail_fan_inversion", 1))
    max_fan_violation_now = int(gate.get("max_fan_violation_now", 1))

    # --- optional boosters ---
    min_fan_spread_slope_bull = float(gate.get("min_fan_spread_slope_bull", -1e9))
    max_fan_spread_slope_bear = float(gate.get("max_fan_spread_slope_bear", 1e9))
    min_tail_momentum = float(gate.get("min_tail_momentum", 0.0))
    min_seg_momentum = float(gate.get("min_seg_momentum", 0.0))

    # --- final thresholds ---
    bull_threshold = float(gate.get("score_bull_threshold", 0.70))
    bear_threshold = float(gate.get("score_bear_threshold", 0.70))

    # --- read features ---
    segment_valid = _i(row, "segment_valid", 0)
    segment_len = _i(row, "segment_len_epochs", 0)
    tail_len = _i(row, "tail_len_epochs", 0)

    anchor_is_peak = _i(row, "anchor_is_peak", 0)
    anchor_is_valley = _i(row, "anchor_is_valley", 0)

    seg_dir_up = _i(row, "segment_dir_up", 0)
    seg_dir_down = _i(row, "segment_dir_down", 0)

    seg_g83_slope = _f(row, "seg_g83_slope")
    tail_g83_slope = _f(row, "tail_g83_slope")
    tail_g83_r2 = _f(row, "tail_g83_r2")
    tail_plateau = _f(row, "tail_plateau_score")
    tail_fan_inversion = _i(row, "tail_fan_inversion_count", 0)
    fan_violation_now = _i(row, "fan_order_violation_now", 0)

    fan_spread_slope = _f(row, "fan_spread_slope")
    tail_momentum = _f(row, "tail_momentum_score")
    seg_momentum = _f(row, "seg_momentum_score")

    # keep some older row-wise features as boosters, not core
    close_loc = _f(row, "close_loc", 0.5)
    body = _f(row, "body", 0.0)
    fan_order_score = _f(row, "fan_order_score", 0.0)
    tail_ret_3 = _f(row, "tail_ret_3", 0.0)
    tail_ret_5 = _f(row, "tail_ret_5", 0.0)

    fail_reasons: list[str] = []

    # --------------------------------------------------
    # Global segment usability gate
    # --------------------------------------------------
    if require_segment_valid and segment_valid != 1:
        return StructureSignal(
            signal="Skip",
            score=0.0,
            template="segment_invalid",
            reason=f"segment_invalid=1|segment_valid={segment_valid}|segment_len={segment_len}|tail_len={tail_len}",
        )

    if segment_len < min_segment_len:
        return StructureSignal(
            signal="Skip",
            score=0.0,
            template="segment_too_short",
            reason=f"segment_too_short|segment_len={segment_len}|min_segment_len={min_segment_len}",
        )

    if tail_len < min_tail_len:
        return StructureSignal(
            signal="Skip",
            score=0.0,
            template="tail_too_short",
            reason=f"tail_too_short|tail_len={tail_len}|min_tail_len={min_tail_len}",
        )

    # --------------------------------------------------
    # Bull continuation core gate
    # --------------------------------------------------
    bull_core_ok = True
    bull_core_fails: list[str] = []

    if anchor_is_valley != 1:
        bull_core_ok = False
        bull_core_fails.append("anchor_not_valley")
    if seg_dir_up != 1:
        bull_core_ok = False
        bull_core_fails.append("segment_not_up")
    if seg_g83_slope <= min_seg_slope:
        bull_core_ok = False
        bull_core_fails.append("seg_slope_not_pos")
    if tail_g83_slope <= min_tail_slope:
        bull_core_ok = False
        bull_core_fails.append("tail_slope_not_pos")
    if tail_g83_r2 < min_tail_r2:
        bull_core_ok = False
        bull_core_fails.append("tail_r2_low")
    if tail_plateau > max_tail_plateau:
        bull_core_ok = False
        bull_core_fails.append("tail_plateau_high")
    if tail_fan_inversion > max_tail_fan_inversion:
        bull_core_ok = False
        bull_core_fails.append("tail_fan_inversion_high")
    if fan_violation_now > max_fan_violation_now:
        bull_core_ok = False
        bull_core_fails.append("fan_violation_now_high")

    bull_score = 0.0
    bull_boosters: list[str] = []

    if bull_core_ok:
        bull_score = 0.72  # hard gate passes
        if fan_spread_slope >= min_fan_spread_slope_bull:
            bull_score += 0.05
            bull_boosters.append("fan_spread_support")
        if tail_momentum >= min_tail_momentum:
            bull_score += 0.05
            bull_boosters.append("tail_momentum_support")
        if seg_momentum >= min_seg_momentum:
            bull_score += 0.03
            bull_boosters.append("seg_momentum_support")
        if fan_order_score > 0:
            bull_score += 0.03
            bull_boosters.append("fan_order_support")
        if close_loc > 0.50 and body > 0:
            bull_score += 0.02
            bull_boosters.append("candle_support")
        if tail_ret_3 > 0 and tail_ret_5 > 0:
            bull_score += 0.02
            bull_boosters.append("tail_ret_support")

    # --------------------------------------------------
    # Bear continuation core gate
    # --------------------------------------------------
    bear_core_ok = True
    bear_core_fails: list[str] = []

    if anchor_is_peak != 1:
        bear_core_ok = False
        bear_core_fails.append("anchor_not_peak")
    if seg_dir_down != 1:
        bear_core_ok = False
        bear_core_fails.append("segment_not_down")
    if seg_g83_slope >= -min_seg_slope:
        bear_core_ok = False
        bear_core_fails.append("seg_slope_not_neg")
    if tail_g83_slope >= -min_tail_slope:
        bear_core_ok = False
        bear_core_fails.append("tail_slope_not_neg")
    if tail_g83_r2 < min_tail_r2:
        bear_core_ok = False
        bear_core_fails.append("tail_r2_low")
    if tail_plateau > max_tail_plateau:
        bear_core_ok = False
        bear_core_fails.append("tail_plateau_high")
    if tail_fan_inversion > max_tail_fan_inversion:
        bear_core_ok = False
        bear_core_fails.append("tail_fan_inversion_high")
    if fan_violation_now > max_fan_violation_now:
        bear_core_ok = False
        bear_core_fails.append("fan_violation_now_high")

    bear_score = 0.0
    bear_boosters: list[str] = []

    if bear_core_ok:
        bear_score = 0.72
        if fan_spread_slope <= max_fan_spread_slope_bear:
            bear_score += 0.05
            bear_boosters.append("fan_spread_support")
        if tail_momentum >= min_tail_momentum:
            bear_score += 0.05
            bear_boosters.append("tail_momentum_support")
        if seg_momentum >= min_seg_momentum:
            bear_score += 0.03
            bear_boosters.append("seg_momentum_support")
        if fan_order_score < 0:
            bear_score += 0.03
            bear_boosters.append("fan_order_support")
        if close_loc < 0.50 and body < 0:
            bear_score += 0.02
            bear_boosters.append("candle_support")
        if tail_ret_3 < 0 and tail_ret_5 < 0:
            bear_score += 0.02
            bear_boosters.append("tail_ret_support")

    # --------------------------------------------------
    # Decision
    # --------------------------------------------------
    if bull_score >= bull_threshold and bull_score > bear_score:
        return StructureSignal(
            signal="Bull",
            score=bull_score,
            template="segment_bull_continuation",
            reason=(
                f"bull_score={bull_score:.3f}|bear_score={bear_score:.3f}"
                f"|segment_valid={segment_valid}|segment_len={segment_len}|tail_len={tail_len}"
                f"|anchor=VALLEY|seg_dir=UP"
                f"|seg_slope={seg_g83_slope:.6f}|tail_slope={tail_g83_slope:.6f}|tail_r2={tail_g83_r2:.6f}"
                f"|tail_plateau={tail_plateau:.6f}|tail_fan_inv={tail_fan_inversion}|fan_violation_now={fan_violation_now}"
                f"|boosters={','.join(bull_boosters) if bull_boosters else 'none'}"
            ),
        )

    if bear_score >= bear_threshold and bear_score > bull_score:
        return StructureSignal(
            signal="Bear",
            score=bear_score,
            template="segment_bear_continuation",
            reason=(
                f"bull_score={bull_score:.3f}|bear_score={bear_score:.3f}"
                f"|segment_valid={segment_valid}|segment_len={segment_len}|tail_len={tail_len}"
                f"|anchor=PEAK|seg_dir=DOWN"
                f"|seg_slope={seg_g83_slope:.6f}|tail_slope={tail_g83_slope:.6f}|tail_r2={tail_g83_r2:.6f}"
                f"|tail_plateau={tail_plateau:.6f}|tail_fan_inv={tail_fan_inversion}|fan_violation_now={fan_violation_now}"
                f"|boosters={','.join(bear_boosters) if bear_boosters else 'none'}"
            ),
        )

    return StructureSignal(
        signal="Skip",
        score=max(bull_score, bear_score),
        template="segment_skip",
        reason=(
            f"bull_score={bull_score:.3f}|bear_score={bear_score:.3f}"
            f"|segment_valid={segment_valid}|segment_len={segment_len}|tail_len={tail_len}"
            f"|bull_core_ok={int(bull_core_ok)}|bear_core_ok={int(bear_core_ok)}"
            f"|bull_core_fails={','.join(bull_core_fails) if bull_core_fails else 'none'}"
            f"|bear_core_fails={','.join(bear_core_fails) if bear_core_fails else 'none'}"
            f"|seg_slope={seg_g83_slope:.6f}|tail_slope={tail_g83_slope:.6f}|tail_r2={tail_g83_r2:.6f}"
            f"|tail_plateau={tail_plateau:.6f}|tail_fan_inv={tail_fan_inversion}|fan_violation_now={fan_violation_now}"
        ),
    )