from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from main.engine.process.core.DB_process_calc import (
    attach_v21_live_bundle,
    build_td_features_for_model,
)
from main.engine.process.models.v21_live.model_runner_v21_live import run_model_v21_live


@dataclass
class TrendDecision:
    trend: str = "Neutral"
    confidence: float = 0.0
    model: str = "v21_live"
    reason: str = ""
    notes: str = ""
    scores: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    calc: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)


def _public_scores_from_result(result: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    scores_dbg = (diagnostics.get("scores") or {}) if isinstance(diagnostics, dict) else {}
    scorecard = (diagnostics.get("scorecard") or {}) if isinstance(diagnostics, dict) else {}
    out = {
        "neutral": scores_dbg.get("neutral_score", result.get("neutral_score")),
        "bull": scores_dbg.get("bull_score", result.get("bull_score")),
        "bear": scores_dbg.get("bear_score", result.get("bear_score")),
        "reversal": scores_dbg.get("reversal_score", result.get("reversal_score")),
        "decision": scores_dbg.get("decision_score"),
        "decision_norm": scores_dbg.get("decision_score_norm"),
        "bull_continuation": scorecard.get("bull_continuation", scores_dbg.get("bull_continuation")),
        "bear_continuation": scorecard.get("bear_continuation", scores_dbg.get("bear_continuation")),
        "bull_reversal": scorecard.get("bull_reversal", scores_dbg.get("bull_reversal")),
        "bear_reversal": scorecard.get("bear_reversal", scores_dbg.get("bear_reversal")),
        "branch": scorecard.get("branch"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _public_features_from_td(td_features: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    gauss = (td_features.get("gauss") or {}) if isinstance(td_features, dict) else {}
    slopes = (gauss.get("slopes") or {}) if isinstance(gauss, dict) else {}
    curvature = (gauss.get("curvature") or {}) if isinstance(gauss, dict) else {}
    mapped = (diagnostics.get("mapped_features") or {}) if isinstance(diagnostics, dict) else {}

    features = {
        "slope_s8": slopes.get("s8"),
        "slope_s23": slopes.get("s23"),
        "slope_s38": slopes.get("s38"),
        "slope_s53": slopes.get("s53"),
        "slope_s68": slopes.get("s68"),
        "slope_s83": slopes.get("s83"),
        "curvature_s23": curvature.get("s23"),
        "curvature_s53": curvature.get("s53"),
        "px_mid_8": mapped.get("px_mid_8"),
        "px_mid_23": mapped.get("px_mid_23"),
        "zpos_83": mapped.get("zpos_83"),
        "contracting_count": mapped.get("contracting_count"),
        "reversal_stage": mapped.get("reversal_stage"),
        "same_sign_exhaustion": mapped.get("same_sign_exhaustion"),
        "msbc_continuity_mean": mapped.get("msbc_continuity_mean"),
        "msbc_override_count": mapped.get("msbc_override_count"),
        "front_hook_active": mapped.get("front_hook_active"),
    }
    return {k: v for k, v in features.items() if v is not None}


def _public_calc_from_diagnostics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    flags = diagnostics.get("flags") or {}
    scores = diagnostics.get("scores") or {}
    validation = diagnostics.get("validation") or {}
    scorecard = diagnostics.get("scorecard") or {}

    vote_raw = scores.get("decision_score")
    vote_norm = scores.get("decision_score_norm")

    guard_fired = bool(validation.get("guard_fired", False))
    guard = {
        "name": validation.get("guard_name", "SRC_CONTRACT_GUARD"),
        "fired": guard_fired,
        "note": validation.get("note", "ok"),
        "reason": validation.get("reason", "VALID"),
        "debug": {
            "probe_warn": bool(flags.get("collapse_watch")),
            "probe_mismatch": bool(flags.get("mid_contradicts_fast")),
            "collapse_ok": not bool(flags.get("collapse_watch")),
            "disorder_ok": float(scorecard.get("structure_disorder_ratio", 0.0)) <= 0.50,
            "ep_sign": diagnostics.get("reason", "?"),
            "pr_sign": scorecard.get("transfer_dir", "?"),
            "flip_watch": bool(flags.get("same_sign_bear_exhaustion_reversal_risk") or flags.get("same_sign_bull_exhaustion_downside_reversion_risk")),
            "fast_collapse": bool(flags.get("collapse_watch")),
            "d_abs_slope_tail": vote_norm,
            "s0_dW_norm": scorecard.get("front_hook_score"),
            "s0_eff_order": validation.get("coverage_overall"),
        },
    }

    return {
        "branch": flags.get("branch"),
        "decision_score": vote_raw,
        "decision_score_norm": vote_norm,
        "vote": {
            "vote_raw": vote_raw,
            "vote_norm": vote_norm,
        },
        "guardrail": guard,
    }


def calculate_trend(
    curr_epoch: int,
    next_epoch: int,
    windows,
    per_sigma_hist,
    config: Dict[str, Any],
    model_path: Optional[str] = None,
    hyst_obj: Optional[Dict[str, Any]] = None,
):
    """
    Live process-compatible trend entrypoint.

    IMPORTANT:
    Keep this signature compatible with DB_process_orchestrator.py.
    """

    try:
        hyst_obj = hyst_obj if isinstance(hyst_obj, dict) else {}

        td_features = build_td_features_for_model(
            curr_epoch=curr_epoch,
            next_epoch=next_epoch,
            windows=windows,
            per_sigma_hist=per_sigma_hist,
            config=config,
            hyst_obj=hyst_obj,
            model_path=model_path,
        )

        src_bundle = attach_v21_live_bundle(
            td_features,
            curr_epoch=curr_epoch,
            next_epoch=next_epoch,
            windows=windows,
            per_sigma_hist=per_sigma_hist,
            config=config,
            hyst_obj=hyst_obj,
        )
        src_summary = src_bundle.get("summary") if isinstance(src_bundle.get("summary"), dict) else {}

        result = run_model_v21_live(td_features, config=config)
        diagnostics = result.get("diagnostics", {}) or {}

        if src_summary:
            diagnostics["v21_live_src"] = src_summary
        if src_bundle.get("source"):
            diagnostics["v21_live_src_source"] = str(src_bundle.get("source"))
        if src_bundle.get("error"):
            diagnostics["v21_live_src_error"] = str(src_bundle.get("error"))

        public_scores = _public_scores_from_result(result, diagnostics)
        public_features = _public_features_from_td(td_features, diagnostics)
        diagnostics.setdefault("features", public_features)
        calc_dbg = _public_calc_from_diagnostics(diagnostics)

        return TrendDecision(
            trend=str(result.get("trend", "Neutral")),
            confidence=float(result.get("confidence", 0.0) or 0.0),
            model=str(result.get("model", "v21_live")),
            reason=str(result.get("reason", "")),
            notes=str(result.get("reason", "")),
            scores=public_scores,
            features=public_features,
            calc=calc_dbg,
            raw=((diagnostics.get("scores") or {}).get("decision_score") if isinstance(diagnostics.get("scores"), dict) else None),
            extras=diagnostics,
        )

    except Exception as e:
        return TrendDecision(
            trend="Neutral",
            confidence=0.0,
            model="v21_live",
            reason="model_error",
            notes="model_error",
            extras={"error": str(e)},
        )
