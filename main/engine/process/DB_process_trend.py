# main/engine/process/DB_process_trend.py

from __future__ import annotations

from datetime import datetime
import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional

from main.engine.process.DB_process_types import TrendDecision
from main.engine.process.DB_process_feature_catalog import build_feature_catalog
from main.engine.process.DB_process_calc import run_enabled_models

log = logging.getLogger(__name__)


def _fallback(model_id: str = "trend_method_v1_0") -> Dict[str, Any]:
    return {
        "model_id": model_id,
        "trend": "Neutral",
        "confidence": 1.0,
        "score": 0.0,
        "reason": "model_error",
        "debug": {"error": "fallback"},
        "raw_features_used": {},
    }

def calculate_trend(
        curr_epoch: int,
        next_epoch: int,
        windows,
        per_sigma_full: Dict[int, Dict[str, Any]],
        config: Dict[str, Any],
        model_path: Optional[str] = None,
        hyst_obj: Optional[Any] = None,
) -> TrendDecision:
    """Runs all enabled trend_method_v1_* models and returns a primary decision for legacy flow."""
    try:
        features = build_feature_catalog(
            timing=windows if hasattr(windows, "curr_epoch") else type("T", (), {"curr_epoch": curr_epoch, "next_epoch": next_epoch, "dt_curr": datetime.now()})(),
            per_sigma_hist=per_sigma_full,
            config=config,
            hyst_obj=hyst_obj if isinstance(hyst_obj, dict) else {},
        )

        now = datetime.now()
        payload = run_enabled_models(
            epoch=int(curr_epoch),
            next_epoch=int(next_epoch),
            timestamp=now.isoformat(),
            features=features,
        )
        model_results = payload.get("models", []) or []
        if not model_results:
            model_results = [_fallback()]

        for result in model_results:
            log.info(
                "%s - Epoch data from %s → Predict Next Epoch %s | trend=%s | confidence=%.3f | model=%s",
                now.strftime("%I:%M:%S %p"),
                curr_epoch,
                next_epoch,
                result.get("trend", "Neutral"),
                float(result.get("confidence", 1.0) or 1.0),
                result.get("model_id", "trend_method_v1_0"),
            )

        primary = model_results[0]
        td = TrendDecision(
            trend=str(primary.get("trend", "Neutral")),
            confidence=float(primary.get("confidence", 1.0) or 1.0),
            model=str(primary.get("model_id", "trend_method_v1_0")),
            notes=str(primary.get("reason", "")),
        )
        setattr(td, "extras", {
            "primary": primary,
            "all_model_predictions": model_results,
            "features": features,
            "model_predictions_payload": payload,
        })
        return td



def _save_trend_out_json(td: TrendDecision,
                         timing: Dict[str, Any],
                         config: Dict[str, Any],
                         slog: Any = None) -> Optional[str]:
    """
    Save TrendDecision once to TREND_OUT_JSON if configured.
    Returns path written, or None if skipped.
    """
    out_path = config.get("TREND_OUT_JSON")
    if not isinstance(out_path, str) or not out_path.strip():
        return None

    try:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "decision_time": timing.get("decision_time") or timing.get("decision_time_str"),
            "epoch_analyze": timing.get("epoch_analyze"),
            "epoch_predict": timing.get("epoch_predict"),
            "trend": getattr(td, "trend", "Neutral"),
            "confidence": float(getattr(td, "confidence", 0.0) or 0.0),
            "model": getattr(td, "model", ""),
            "notes": getattr(td, "notes", ""),
            "extras": getattr(td, "extras", None),
        }

        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(p)

    except Exception as e:
        if slog:
            slog.error(f"[trend_save] failed writing TREND_OUT_JSON: {e}")
        return None

