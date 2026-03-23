from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier

from .base_model import BaseEpochModel


class XGBFallbackEpochModel(BaseEpochModel):
    """
    Starter fallback using sklearn HistGradientBoosting so you can run without
    adding xgboost immediately. Later, swap this for true XGBoost or LightGBM.
    """
    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=250,
            random_state=42,
        )

    def fit(self, x, y):
        self.model.fit(x, y)
        self._is_fit = True

    def predict_proba(self, x):
        return self.model.predict_proba(x)