# main/engine/process/DB_process_trend.py

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

from main.engine.process.DB_process_types import TrendDecision
from main.engine.process.printing.DB_process_printing_hyst import compute_hysteresis_features
from main.engine.process.printing.DB_process_SectionLogger import get_section_logger

DEFAULT_MODEL_FILE = "method_hyster_v1.0.json"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / DEFAULT_MODEL_FILE


def _decision_from_hyst(hyst_obj: Dict[str, Any]) -> tuple[str, float, str]:
    ep = (hyst_obj or {}).get("episode", {}) or {}
    probe = (hyst_obj or {}).get("probe", {}) or {}

    sign_now = int(ep.get("sign_now", 0) or 0)
    mode = str(ep.get("mode", "DEFAULT"))
    flip_watch = bool(probe.get("flip_watch", False))
    fast_collapse = bool(probe.get("fast_collapse", False))

    if sign_now > 0:
        trend = "Bull"
    elif sign_now < 0:
        trend = "Bear"
    else:
        trend = "Neutral"

    conf = 0.45 if trend == "Neutral" else 0.62
    if mode == "SLOW_SWITCH":
        conf = min(1.0, conf + 0.08)
    if flip_watch or fast_collapse:
        conf = max(0.0, conf - 0.12)

    reason = f"HYST_SIGN={sign_now}|MODE={mode}|FLIP_WATCH={int(flip_watch)}|FAST_COLLAPSE={int(fast_collapse)}"
    return trend, conf, reason


def calculate_trend(
    curr_epoch: int,
    next_epoch: int,
    windows,
    per_sigma_full: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
    model_path: Optional[str] = None,
) -> TrendDecision:
    slog = get_section_logger(logging.getLogger(__name__), config)

    model_cfg = (config.get("MODEL_hyster", {}) if isinstance(config, dict) else {}) or {}
    enabled = bool(model_cfg.get("ENABLED", False))
    model_id = str(model_cfg.get("MODEL_ID", DEFAULT_MODEL_FILE))

    if not enabled:
        reason = "MODEL_DISABLED"
        slog.TREND_DECISION(f"[trend] MODEL_hyster disabled; returning Neutral | reason={reason}")
        td = TrendDecision(trend="Neutral", confidence=0.0, model=model_id, notes=reason)
        setattr(td, "model_id", model_id)
        setattr(td, "reason", reason)
        setattr(td, "scores", {"neutral": 1.0, "bull": 0.0, "bear": 0.0, "reversal": 0.0})
        setattr(td, "features", {})
        setattr(td, "raw", "Neutral")
        setattr(td, "extras", {"reason": reason})
        return td

    model_candidate = Path(model_path) if model_path else None
    resolved_model = model_candidate if model_candidate and model_candidate.name == model_id else (DEFAULT_MODEL_PATH.parent / model_id)

    if not resolved_model.exists():
        reason = "MODEL_MISSING"
        slog.TREND_DECISION(f"[trend] missing hysteresis model; returning Neutral | reason={reason} | model={resolved_model}")
        slog.TREND_FEATURES(f"[trend_debug] model_id={model_id} | resolved={resolved_model} | exists=0")
        td = TrendDecision(trend="Neutral", confidence=0.0, model=model_id, notes=reason)
        setattr(td, "model_id", model_id)
        setattr(td, "reason", reason)
        setattr(td, "scores", {"neutral": 1.0, "bull": 0.0, "bear": 0.0, "reversal": 0.0})
        setattr(td, "features", {})
        setattr(td, "raw", "Neutral")
        setattr(td, "extras", {"reason": reason, "model_path": str(resolved_model)})
        return td

    decision_dt = getattr(windows, "full_end", None) or datetime.now()
    hyst_obj = compute_hysteresis_features(
        per_sigma_hist=per_sigma_full,
        decision_dt=decision_dt,
        lookback_seconds=3600,
        tail_seconds=120,
        align_tol_seconds=3.0,
    )

    if (hyst_obj.get("meta", {}) or {}).get("skipped"):
        reason = f"HYST_SKIPPED:{hyst_obj.get('meta', {}).get('skip_reason', 'unknown')}"
        trend, confidence = "Neutral", 0.0
    else:
        trend, confidence, reason = _decision_from_hyst(hyst_obj)

    scores = {
        "neutral": 1.0 - confidence if trend == "Neutral" else 0.0,
        "bull": confidence if trend == "Bull" else 0.0,
        "bear": confidence if trend == "Bear" else 0.0,
        "reversal": 0.0,
    }
    features = {
        "hyst_sign_now": int((hyst_obj.get("episode", {}) or {}).get("sign_now", 0) or 0),
        "hyst_elapsed_seconds": float((hyst_obj.get("episode", {}) or {}).get("elapsed_seconds") or 0.0),
        "hyst_probe_flip_watch": int(bool((hyst_obj.get("probe", {}) or {}).get("flip_watch", False))),
        "hyst_probe_fast_collapse": int(bool((hyst_obj.get("probe", {}) or {}).get("fast_collapse", False))),
    }

    slog.TREND_DECISION(
        f"[trend] hysteresis decision | epoch={curr_epoch}->{next_epoch} | trend={trend} | conf={confidence:.3f} | model={model_id} | reason={reason}"
    )
    slog.TREND_SCORES(
        f"[trend_scores] neutral={scores['neutral']:.4f} | bull={scores['bull']:.4f} | bear={scores['bear']:.4f} | reversal={scores['reversal']:.4f}"
    )
    slog.TREND_FEATURES(
        f"[trend_features] sign={features['hyst_sign_now']} | elapsed={features['hyst_elapsed_seconds']:.1f} | "
        f"flip_watch={features['hyst_probe_flip_watch']} | fast_collapse={features['hyst_probe_fast_collapse']}"
    )

    td = TrendDecision(trend=trend, confidence=float(confidence), model=model_id, notes=reason)
    setattr(td, "model_id", model_id)
    setattr(td, "reason", reason)
    setattr(td, "scores", scores)
    setattr(td, "features", features)
    setattr(td, "raw", trend)
    setattr(td, "extras", {"reason": reason, "scores": scores, "features": features})
    return td
