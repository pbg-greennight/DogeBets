from __future__ import annotations

from typing import Any, Dict


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def build_v21_validation_summary(src_audit: dict | None, ctx: dict | None = None, config: dict | None = None) -> dict:
    src_audit = _as_dict(src_audit)
    coverage = _as_dict(src_audit.get("family_coverage"))
    config = _as_dict(config)

    coverage_floor = _f(config.get("V21_LIVE_COVERAGE_FLOOR"), 0.85)
    coverage_overall = _f(coverage.get("overall"), 0.0)
    missing_required = list(src_audit.get("missing_required") or [])
    missing_optional = list(src_audit.get("missing_optional") or [])

    coverage_guard = coverage_overall < coverage_floor
    guard_fired = bool(missing_required) or coverage_guard

    note = "ok"
    reason = "VALID"
    if missing_required:
        note = "missing_required"
        reason = "SRC_CONTRACT_MISSING_REQUIRED"
    elif coverage_guard:
        note = "low_coverage"
        reason = "SRC_CONTRACT_LOW_COVERAGE"

    return {
        "is_valid": bool(src_audit.get("is_valid", False)),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "missing_required_count": len(missing_required),
        "missing_optional_count": len(missing_optional),
        "coverage_overall": coverage_overall,
        "coverage_floor": coverage_floor,
        "coverage_guard": bool(coverage_guard),
        "guard_fired": bool(guard_fired),
        "guard_name": "SRC_CONTRACT_GUARD",
        "note": note,
        "reason": reason,
        "coverage_meta": _f(coverage.get("meta"), 0.0),
        "coverage_msbc": _f(coverage.get("msbc"), 0.0),
        "coverage_gcs": _f(coverage.get("gcs"), 0.0),
        "coverage_csd": _f(coverage.get("csd"), 0.0),
        "coverage_dcsd": _f(coverage.get("dcsd"), 0.0),
        "coverage_gbc": _f(coverage.get("gbc"), 0.0),
        "coverage_hyst": _f(coverage.get("hyst"), 0.0),
        "alias_applied_count": len(src_audit.get("alias_applied") or []),
        "null_count": int(src_audit.get("null_count") or 0),
    }


def build_v21_scorecard(ctx: dict | None, score_obj: dict | None, decision: dict | None, config: dict | None = None) -> dict:
    ctx = _as_dict(ctx)
    score_obj = _as_dict(score_obj)
    decision = _as_dict(decision)

    continuation_score = _f(score_obj.get("continuation_score"), 0.0)
    reversal_score = _f(score_obj.get("reversal_score"), 0.0)
    decision_score = _f(decision.get("decision_score"), 0.0)
    neutral_score = _f(decision.get("neutral_score"), 0.0)

    bull_continuation = max(0.0, continuation_score)
    bear_continuation = max(0.0, -continuation_score)
    bull_reversal = max(0.0, reversal_score)
    bear_reversal = max(0.0, -reversal_score)

    directional_total = abs(continuation_score) + abs(reversal_score) + neutral_score + 1e-9
    decision_score_norm = decision_score / directional_total

    return {
        "branch": decision.get("branch"),
        "trend": decision.get("trend"),
        "bull_continuation": bull_continuation,
        "bear_continuation": bear_continuation,
        "bull_reversal": bull_reversal,
        "bear_reversal": bear_reversal,
        "bull_total": bull_continuation + bull_reversal,
        "bear_total": bear_continuation + bear_reversal,
        "neutral": neutral_score,
        "decision_score": decision_score,
        "decision_score_norm": decision_score_norm,
        "continuation_adjust": _f(score_obj.get("continuation_adjust"), 0.0),
        "reversal_adjust": _f(score_obj.get("reversal_adjust"), 0.0),
        "structure_continuation_support": _f(score_obj.get("structure_continuation_support"), 0.0),
        "structure_reversal_support": _f(score_obj.get("structure_reversal_support"), 0.0),
        "structure_disorder_ratio": _f(score_obj.get("structure_disorder_ratio"), 0.0),
        "front_hook_score": _f(score_obj.get("front_hook_score"), 0.0),
        "transfer_depth": _f(ctx.get("transfer_depth"), 0.0),
        "transfer_dir": ctx.get("transfer_dir"),
        "mapped_missing_count": len(ctx.get("mapped_missing") or []),
        "src_row_available": bool(ctx.get("src_row_available")),
        "mapped_context_active": bool(ctx.get("mapped_context_active")),
    }


def build_v21_debug_report(src_row: dict | None, ctx: dict | None, score_obj: dict | None, decision: dict | None, config: dict | None = None) -> dict:
    ctx = _as_dict(ctx)
    score_obj = _as_dict(score_obj)
    decision = _as_dict(decision)
    validation = build_v21_validation_summary(ctx.get("src_audit"), ctx=ctx, config=config)
    scorecard = build_v21_scorecard(ctx, score_obj, decision, config=config)
    return {
        "validation": validation,
        "scorecard": scorecard,
    }


__all__ = [
    "build_v21_debug_report",
    "build_v21_scorecard",
    "build_v21_validation_summary",
]
