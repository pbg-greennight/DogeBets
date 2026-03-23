from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from main.engine.process.core.DB_process_calc import build_td_features_for_model
from main.engine.process.models.v21_live.model_runner_v21_live import run_model_v21_live


@dataclass
class TrendDecision:
    trend: str = "Neutral"
    confidence: float = 0.0
    model: str = "v21_live"
    notes: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


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
        td_features = build_td_features_for_model(
            curr_epoch=curr_epoch,
            next_epoch=next_epoch,
            windows=windows,
            per_sigma_hist=per_sigma_hist,
            config=config,
            hyst_obj=hyst_obj or {},
            model_path=model_path,
        )

        # ---------------------------------------------------------------------
        # v21 live src_row wiring
        # ---------------------------------------------------------------------
        src_row = {}
        try:
            # Preferred source: orchestrator stashed it on windows
            src_row = getattr(windows, "v21_live_src_row", None) or {}
        except Exception:
            src_row = {}

        if not src_row:
            try:
                # Fallback: orchestrator also placed it inside hyst_obj when available
                if isinstance(hyst_obj, dict):
                    src_row = hyst_obj.get("v21_live_src_row") or {}
            except Exception:
                src_row = {}

        if not src_row:
            try:
                # Last fallback: config stash
                if isinstance(config, dict):
                    src_row = config.get("_V21_LIVE_SRC_ROW") or {}
            except Exception:
                src_row = {}

        if src_row:
            td_features["src_row"] = src_row
            td_features["live_src_row"] = src_row
        # ---------------------------------------------------------------------

        result = run_model_v21_live(td_features, config=config)

        return TrendDecision(
            trend=str(result.get("trend", "Neutral")),
            confidence=float(result.get("confidence", 0.0) or 0.0),
            model=str(result.get("model", "v21_live")),
            notes=str(result.get("reason", "")),
            extras=result.get("diagnostics", {}) or {},
        )

    except Exception as e:
        return TrendDecision(
            trend="Neutral",
            confidence=1.0,
            model="v21_live",
            notes="model_error",
            extras={"error": str(e)},
        )