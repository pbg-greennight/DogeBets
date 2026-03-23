from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PredictionResult:
    trend: str
    prob_bull: float
    prob_bear: float
    confidence: float
    wager: bool
    reason: str


class BaseEpochModel:
    def __init__(self, model_cfg: dict):
        self.model_cfg = model_cfg
        self.model_id = model_cfg["model_id"]
        self.family = model_cfg.get("family", "unknown")
        self.feature_blocks = model_cfg.get("feature_blocks", [])
        self.gate_cfg = model_cfg.get("gate", {})
        self.feature_cols: list[str] = []
        self._is_fit = False

    def fit(self, x, y):
        raise NotImplementedError

    def predict_proba(self, x):
        raise NotImplementedError

    def apply_gate(self, prob_bull: float, prob_bear: float) -> tuple[bool, str]:
        confidence = max(prob_bull, prob_bear)
        min_conf = float(self.gate_cfg.get("min_confidence", 0.55))
        if confidence < min_conf:
            return False, f"confidence<{min_conf:.2f}"
        return True, "confidence_ok"

    def predict_one(self, x_row) -> PredictionResult:
        probs = self.predict_proba(x_row)[0]
        prob_bear = float(probs[0])
        prob_bull = float(probs[1])
        confidence = max(prob_bull, prob_bear)
        trend = "Bull" if prob_bull >= prob_bear else "Bear"
        wager, reason = self.apply_gate(prob_bull, prob_bear)
        return PredictionResult(
            trend=trend,
            prob_bull=prob_bull,
            prob_bear=prob_bear,
            confidence=confidence,
            wager=wager,
            reason=reason,
        )