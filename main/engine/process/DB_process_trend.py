# main/engine/process/DB_process_trend.py

from __future__ import annotations

from datetime import datetime
import logging
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from main.engine.process.DB_process_types import TrendDecision
from main.engine.process.DB_process_feature_catalog import build_feature_catalog
from main.engine.process.DB_process_calc import run_enabled_models

log = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
TREND_LOG_FILE = BASE_DIR / "ts" / "json" / "DB_rounds_trend.json"


def save_forecast_log(trend_label, confidence, next_epoch, model_version, mode):
    """Persist a trend prediction entry to DB_rounds_trend.json."""
    entry = {
        "timestamp": time.strftime("%m/%d/%Y %I:%M:%S %p"),
        "trend": trend_label,
        "confidence": round(float(confidence), 3),
        "next_epoch": next_epoch,
        "model_version": model_version,
        "mode": mode,
    }

    history = []
    if TREND_LOG_FILE.exists():
        try:
            with TREND_LOG_FILE.open("r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(entry)
    history = history[-2000:]

    TREND_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TREND_LOG_FILE.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def _fallback(model_id: str = "trend_method_v2_0") -> Dict[str, Any]:
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
    """Runs the enabled trend_method_v2_0 model and returns a primary decision."""
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

        primary = model_results[0]
        confidence = float(primary.get("confidence", 1.0) or 1.0)
        model_id = str(primary.get("model_id", "trend_method_v2_0"))

        log.info("Trend Decision (%s)", model_id)
        log.info("Epoch %s → Predict Next Epoch %s", curr_epoch, next_epoch)
        log.info(
            "trend=%s | confidence=%.3f | model=%s",
            primary.get("trend", "Neutral"),
            confidence,
            model_id,
        )

        save_forecast_log(
            trend_label=str(primary.get("trend", "Neutral")),
            confidence=confidence,
            next_epoch=int(next_epoch),
            model_version=model_id,
            mode="DB_process_trend",
        )
        td = TrendDecision(
            trend=str(primary.get("trend", "Neutral")),
            confidence=float(primary.get("confidence", 1.0) or 1.0),
            model=str(primary.get("model_id", "trend_method_v2_0")),
            notes=str(primary.get("reason", "")),
        )
        setattr(td, "extras", {
            "primary": primary,
            "all_model_predictions": model_results,
            "features": features,
            "model_predictions_payload": payload,
        })
        return td
    except Exception as e:
        log.exception("calculate_trend failed: %s", e)
        fallback = _fallback(model_id="trend_method_v2_0")
        save_forecast_log(
            trend_label=str(fallback.get("trend", "Neutral")),
            confidence=float(fallback.get("confidence", 1.0) or 1.0),
            next_epoch=int(next_epoch),
            model_version="trend_method_v2_0",
            mode="DB_process_trend_fallback",
        )
        return TrendDecision(
            trend=str(fallback.get("trend", "Neutral")),
            confidence=float(fallback.get("confidence", 1.0) or 1.0),
            model="trend_method_v1_5",
            notes=str(fallback.get("reason", "model_error")),
        )


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

