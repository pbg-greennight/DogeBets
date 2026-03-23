from __future__ import annotations

from typing import Any

import pandas as pd


def _neutral(reason: str, scores: dict[str, float], debug: dict[str, Any]) -> dict[str, Any]:
    return {
        "trend": "Neutral",
        "confidence": min(0.60, float(scores["neutral_raw"])),
        "reason": reason,
        "scores": scores,
        "debug": debug,
    }


def compute_v21_confidence(trend: str, bull_raw: float, bear_raw: float, separation: float, row: pd.Series) -> float:
    from .common_v21 import clip01

    top_score = bull_raw if trend == "Bull" else bear_raw
    direction_health = (row["v21_bull_continuation_score"] - row["v21_bull_exhaustion_score"]) if trend == "Bull" else (row["v21_bear_continuation_score"] - row["v21_bear_exhaustion_score"])
    direction_health = clip01((direction_health + 1.0) / 2.0)
    return clip01(
        0.35 * top_score
        + 0.25 * clip01((separation - 0.08) / 0.25)
        + 0.15 * (1.0 - row["v21_conflict_score"])
        + 0.10 * (1.0 - row["v21_compression_trap_score"])
        + 0.15 * direction_health
    )


def trend_method_v2_1_rules(row: pd.Series, config: dict) -> dict[str, Any]:
    from .common_v21 import clip01

    bull_raw = clip01(
        0.58 * row["v21_bull_continuation_score"]
        + 0.27 * row["v21_bear_exhaustion_score"]
        + 0.10 * row["v21_reversal_to_bull_bonus"]
        - 0.15 * row["v21_conflict_score"]
        - 0.10 * row["v21_compression_trap_score"]
        - 0.10 * row["v21_bull_invalidators"]
    )
    bear_raw = clip01(
        0.58 * row["v21_bear_continuation_score"]
        + 0.27 * row["v21_bull_exhaustion_score"]
        + 0.10 * row["v21_reversal_to_bear_bonus"]
        - 0.15 * row["v21_conflict_score"]
        - 0.10 * row["v21_compression_trap_score"]
        - 0.10 * row["v21_bear_invalidators"]
    )
    neutral_raw = clip01(
        0.32 * (1.0 - max(row["v21_bull_continuation_score"], row["v21_bear_continuation_score"]))
        + 0.28 * row["v21_conflict_score"]
        + 0.22 * row["v21_compression_trap_score"]
        + 0.18 * (1.0 - abs(bull_raw - bear_raw))
    )

    scores = {
        "bull_cont": float(row["v21_bull_continuation_score"]),
        "bear_cont": float(row["v21_bear_continuation_score"]),
        "bull_exhaust": float(row["v21_bull_exhaustion_score"]),
        "bear_exhaust": float(row["v21_bear_exhaustion_score"]),
        "conflict": float(row["v21_conflict_score"]),
        "trap": float(row["v21_compression_trap_score"]),
        "bull_raw": float(bull_raw),
        "bear_raw": float(bear_raw),
        "neutral_raw": float(neutral_raw),
        "separation": float(abs(bull_raw - bear_raw)),
    }
    debug = {"passed": [], "failed": [], "features": {}}

    if row["v21_conflict_score"] >= config["v21"]["conflict_neutral"] and max(row["v21_bull_continuation_score"], row["v21_bear_continuation_score"]) <= 0.64:
        debug["failed"].append("conflict_neutral_gate")
        return _neutral("NEU_HIGH_CONFLICT", scores, debug)
    if row["v21_compression_trap_score"] >= config["v21"]["trap_neutral"] and row["v21_msbc_fast_slow_conflict"] >= config["v21"]["fast_slow_warn"]:
        debug["failed"].append("compression_trap_gate")
        return _neutral("NEU_COMPRESSION_TRAP", scores, debug)
    if row["v21_bull_continuation_score"] < config["v21"]["continuation_min_soft"] and row["v21_bear_continuation_score"] < config["v21"]["continuation_min_soft"]:
        debug["failed"].append("weak_both_continuation")
        return _neutral("NEU_WEAK_CONTINUATION_BOTH", scores, debug)
    if abs(bull_raw - bear_raw) < config["v21"]["neutral_close_call"] and neutral_raw >= 0.58:
        debug["failed"].append("close_call_gate")
        return _neutral("NEU_CLOSE_CALL", scores, debug)

    top_dir = "Bull" if bull_raw > bear_raw else "Bear"
    top_score = max(bull_raw, bear_raw)
    runner_score = min(bull_raw, bear_raw)
    separation = top_score - runner_score
    scores["separation"] = float(separation)

    valid_bull = bull_raw >= config["v21"]["direction_min"] and separation >= config["v21"]["separation_min"] and row["v21_conflict_score"] < config["v21"]["conflict_block"] and row["v21_compression_trap_score"] < config["v21"]["trap_block"] and row["v21_bull_invalidators"] < 0.62
    valid_bear = bear_raw >= config["v21"]["direction_min"] and separation >= config["v21"]["separation_min"] and row["v21_conflict_score"] < config["v21"]["conflict_block"] and row["v21_compression_trap_score"] < config["v21"]["trap_block"] and row["v21_bear_invalidators"] < 0.62

    if top_dir == "Bull" and not valid_bull:
        debug["failed"].append("bull_validity")
        return _neutral("NEU_BULL_INVALIDATED", scores, debug)
    if top_dir == "Bear" and not valid_bear:
        debug["failed"].append("bear_validity")
        return _neutral("NEU_BEAR_INVALIDATED", scores, debug)

    if top_dir == "Bull":
        if row["v21_bull_continuation_score"] < config["v21"]["continuation_min_hard"] and row["v21_compression_trap_score"] > 0.58 and row["v21_msbc_fast_slow_conflict"] > config["v21"]["fast_slow_warn"]:
            debug["failed"].append("bull_fast_only_countertrend")
            return _neutral("NEU_FAST_ONLY_COUNTERTREND", scores, debug)
        reason = "BULL_CONTINUATION_CLEAN"
    else:
        if row["v21_bear_continuation_score"] < config["v21"]["continuation_min_hard"] and row["v21_bear_exhaustion_score"] > config["v21"]["exhaustion_override"] and row["v21_hyst_frozen_exhaustion"] > 0.58:
            debug["failed"].append("bear_late_run_exhaustion")
            return _neutral("NEU_LATE_RUN_EXHAUSTION", scores, debug)
        reason = "BEAR_CONTINUATION_CLEAN"

    confidence = compute_v21_confidence(top_dir, bull_raw, bear_raw, separation, row)
    return {
        "trend": top_dir,
        "confidence": float(confidence),
        "reason": reason,
        "scores": scores,
        "debug": debug,
    }
