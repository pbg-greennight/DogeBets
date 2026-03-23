from __future__ import annotations

import pandas as pd

from .base_model import BaseEpochModel, PredictionResult
from main.research.epoch_tournament.backtest.structure_rules import generate_structure_signal


class StructureRuleModel(BaseEpochModel):
    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)

    def fit(self, x, y):
        self._is_fit = True

    def predict_proba(self, x):
        raise NotImplementedError("StructureRuleModel uses predict_one() directly.")

    def predict_one(self, x_row) -> PredictionResult:
        if isinstance(x_row, pd.DataFrame):
            row = x_row.iloc[0]
        else:
            row = x_row

        sig = generate_structure_signal(row, self.model_cfg)

        if sig.signal == "Bull":
            prob_bull = float(min(max(sig.score, 0.0), 1.0))
            prob_bear = float(1.0 - prob_bull)
            confidence = prob_bull
            wager = True
            trend = "Bull"

        elif sig.signal == "Bear":
            prob_bear = float(min(max(sig.score, 0.0), 1.0))
            prob_bull = float(1.0 - prob_bear)
            confidence = prob_bear
            wager = True
            trend = "Bear"

        else:
            prob_bull = 0.50
            prob_bear = 0.50
            confidence = float(sig.score)
            wager = False
            trend = "Skip"

        return PredictionResult(
            trend=trend,
            prob_bull=prob_bull,
            prob_bear=prob_bear,
            confidence=confidence,
            wager=wager,
            reason=f"{sig.template}:{sig.reason}",
        )